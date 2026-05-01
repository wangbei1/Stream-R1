import gc
import logging
import csv
import datetime

from utils.dataset import ShardingLMDBDataset, cycle
from utils.dataset import TextDataset
from utils.distributed import EMA_FSDP, fsdp_wrap, fsdp_state_dict, launch_distributed_job
from utils.misc import (
    set_seed,
    merge_dict_list
)
import torch.distributed as dist
from omegaconf import OmegaConf
from model import CausVid, DMD, SiD
import torch
import wandb
import time
import os


class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0
        self._gen_log_accum = []
        self._critic_log_accum = []

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.causal = config.causal
        self.disable_wandb = config.disable_wandb

        self.gradient_accumulation_steps = getattr(config, "gradient_accumulation_steps", 1)
        
        if self.gradient_accumulation_steps > 1:
            self.config.log_iters = self.config.log_iters * self.gradient_accumulation_steps
            self.config.full_training_steps = self.config.full_training_steps * self.gradient_accumulation_steps
            print(f"INFO: Using gradient accumulation with {self.gradient_accumulation_steps} steps. "
                  f"log_iters = {self.config.log_iters}, full_training_steps = {self.config.full_training_steps}")


        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + global_rank)

        if self.is_main_process and not self.disable_wandb:
            wandb.login(host=config.wandb_host, key=config.wandb_key)
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=config.config_name,
                mode="online",
                entity=config.wandb_entity,
                project=config.wandb_project,
                dir=config.wandb_save_dir
            )

        self.output_path = config.logdir

        # Step 2: Initialize the model and optimizer
        if config.distribution_loss == "causvid":
            self.model = CausVid(config, device=self.device)
        elif config.distribution_loss == "dmd":
            self.model = DMD(config, device=self.device)
        elif config.distribution_loss == "sid":
            self.model = SiD(config, device=self.device)
        else:
            raise ValueError("Invalid distribution matching loss")

        # Save pretrained model state_dicts to CPU
        self.fake_score_state_dict_cpu = self.model.fake_score.state_dict()

        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy
        )

        self.model.real_score = fsdp_wrap(
            self.model.real_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.real_score_fsdp_wrap_strategy
        )

        self.model.fake_score = fsdp_wrap(
            self.model.fake_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.fake_score_fsdp_wrap_strategy
        )

        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "text_encoder_cpu_offload", False)
        )

        if not config.no_visualize or config.load_raw_video:
            self.model.vae = self.model.vae.to(
                device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )

        self.critic_optimizer = torch.optim.AdamW(
            [param for param in self.model.fake_score.parameters()
             if param.requires_grad],
            lr=config.lr_critic if hasattr(config, "lr_critic") else config.lr,
            betas=(config.beta1_critic, config.beta2_critic),
            weight_decay=config.weight_decay
        )

        # Step 3: Initialize the dataloader
        if self.config.i2v:
            dataset = ShardingLMDBDataset(config.data_path, max_pair=int(1e8))
        else:
            dataset = TextDataset(config.data_path)
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=8)

        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataset))
        self.dataloader = cycle(dataloader)

        ##############################################################################################################
        # 6. Set up EMA parameter containers
        rename_param = (
            lambda name: name.replace("_fsdp_wrapped_module.", "")
            .replace("_checkpoint_wrapped_module.", "")
            .replace("_orig_mod.", "")
        )
        self.name_to_trainable_params = {}
        for n, p in self.model.generator.named_parameters():
            if not p.requires_grad:
                continue

            renamed_n = rename_param(n)
            self.name_to_trainable_params[renamed_n] = p
        ema_weight = config.ema_weight
        self.generator_ema = None
        if (ema_weight is not None) and (ema_weight > 0.0):
            print(f"Setting up EMA with weight {ema_weight}")
            self.generator_ema = EMA_FSDP(self.model.generator, decay=ema_weight)

        ##############################################################################################################
        # 7. (If resuming) Load the model and optimizer, lr_scheduler, ema's statedicts
        if getattr(config, "generator_ckpt", False):
            print(f"Loading pretrained generator from {config.generator_ckpt}")
            state_dict = torch.load(config.generator_ckpt, map_location="cpu")
            if "generator" in state_dict:
                state_dict = state_dict["generator"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            self.model.generator.load_state_dict(
                state_dict, strict=True
            )

        ##############################################################################################################

        # Let's delete EMA params for early steps to save some computes at training and inference
        if self.step < config.ema_start_step:
            self.generator_ema = None

        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)
        self.previous_time = None

        # Initialize metrics log file (main process only)
        self._log_file = None
        self._log_writer = None
        self._log_records = []
        if self.is_main_process:
            self._init_log_file()

    def _init_log_file(self):
        os.makedirs(self.output_path, exist_ok=True)
        log_path = os.path.join(self.output_path, "train_log.csv")
        self._log_file = open(log_path, "w", newline="", buffering=1)
        fieldnames = [
            "step", "wall_time", "elapsed_sec",
            "generator_loss", "critic_loss",
            "generator_grad_norm", "critic_grad_norm",
            "dmd_gradient_norm",
        ]
        self._log_writer = csv.DictWriter(self._log_file, fieldnames=fieldnames)
        self._log_writer.writeheader()
        self._log_start_time = time.time()
        with open(os.path.join(self.output_path, "train_log.txt"), "w") as f:
            f.write(f"# Experiment: {self.config.config_name}\n")
            f.write(f"# Trainer: score_distillation (DMD, no reward)\n")
            f.write(f"# Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# total_steps={self.config.full_training_steps}  "
                    f"gradient_accumulation={self.config.gradient_accumulation_steps}\n")
            f.write(f"# Columns: step | wall_time | elapsed_sec | generator_loss | critic_loss | "
                    f"generator_grad_norm | critic_grad_norm | dmd_gradient_norm\n")
            f.write("#\n")

    def _write_log(self, step, gen_dict, crit_dict):
        elapsed = time.time() - self._log_start_time
        wall_time = datetime.datetime.now().strftime("%H:%M:%S")

        def _v(d, k):
            val = d.get(k, None)
            if val is None:
                return ""
            if isinstance(val, torch.Tensor):
                val = val.mean().item()
            return f"{val:.6f}"

        row = {
            "step": step,
            "wall_time": wall_time,
            "elapsed_sec": f"{elapsed:.1f}",
            "generator_loss": _v(gen_dict, "generator_loss"),
            "critic_loss": _v(crit_dict, "critic_loss"),
            "generator_grad_norm": _v(gen_dict, "generator_grad_norm"),
            "critic_grad_norm": _v(crit_dict, "critic_grad_norm"),
            "dmd_gradient_norm": _v(gen_dict, "dmdtrain_gradient_norm"),
        }
        self._log_writer.writerow(row)

        with open(os.path.join(self.output_path, "train_log.txt"), "a") as f:
            f.write(
                f"step={step:6d}  t={wall_time}  elapsed={elapsed:7.0f}s  "
                f"gen_loss={row['generator_loss']:>10s}  crit_loss={row['critic_loss']:>10s}  "
                f"dmd_gnorm={row['dmd_gradient_norm']:>10s}\n"
            )

        self._log_records.append({k: (float(v) if v != "" else None) for k, v in row.items()
                                   if k not in ("wall_time",)})

    def _plot_metrics(self):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available, skipping plot generation.")
            return

        if not self._log_records:
            return

        steps = [r["step"] for r in self._log_records]

        def _extract(key):
            vals = [r.get(key) for r in self._log_records]
            pairs = [(s, v) for s, v in zip(steps, vals) if v is not None]
            if not pairs:
                return [], []
            return zip(*pairs)

        metrics = [
            ("generator_loss",      "Generator Loss",      "tab:blue"),
            ("critic_loss",         "Critic Loss",         "tab:orange"),
            ("generator_grad_norm", "Generator Grad Norm", "tab:green"),
            ("critic_grad_norm",    "Critic Grad Norm",    "tab:red"),
            ("dmd_gradient_norm",   "DMD Gradient Norm",   "tab:purple"),
        ]

        active = [(k, lbl, c) for k, lbl, c in metrics if any(r.get(k) is not None for r in self._log_records)]

        n = len(active)
        cols = 2
        rows = (n + 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows))
        axes = axes.flatten() if n > 1 else [axes]

        for ax, (key, label, color) in zip(axes, active):
            xs, ys = _extract(key)
            xs, ys = list(xs), list(ys)
            ax.plot(xs, ys, color=color, linewidth=1.2)
            ax.set_title(label, fontsize=11)
            ax.set_xlabel("Step")
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.3)

        for ax in axes[len(active):]:
            ax.set_visible(False)

        fig.suptitle(f"{self.config.config_name} — Training Curves", fontsize=13, y=1.01)
        plt.tight_layout()
        plot_path = os.path.join(self.output_path, "training_curves.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Training curves saved to {plot_path}")

    def save(self):
        print("Start gathering distributed model states...")
        generator_state_dict = fsdp_state_dict(
            self.model.generator)
        critic_state_dict = fsdp_state_dict(
            self.model.fake_score)

        if self.config.ema_start_step < self.step:
            state_dict = {
                "generator": generator_state_dict,
                "critic": critic_state_dict,
                "generator_ema": self.generator_ema.state_dict(),
            }
        else:
            state_dict = {
                "generator": generator_state_dict,
                "critic": critic_state_dict,
            }

        if self.is_main_process:
            os.makedirs(os.path.join(self.output_path,
                        f"checkpoint_model_{self.step:06d}"), exist_ok=True)
            torch.save(state_dict, os.path.join(self.output_path,
                       f"checkpoint_model_{self.step:06d}", "model.pt"))
            print("Model saved to", os.path.join(self.output_path,
                  f"checkpoint_model_{self.step:06d}", "model.pt"))

    def fwdbwd_one_step(self, batch, train_generator):
        self.model.eval()  # prevent any randomness (e.g. dropout)

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        # Step 1: Get the next batch of text prompts
        text_prompts = batch["prompts"]
        if self.config.i2v:
            clean_latent = None
            image_latent = batch["ode_latent"][:, -1][:, 0:1, ].to(
                device=self.device, dtype=self.dtype)
        else:
            clean_latent = None
            image_latent = None

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        # Step 2: Extract the conditional infos
        with torch.no_grad():
            conditional_dict = self.model.text_encoder(
                text_prompts=text_prompts)

            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size)
                unconditional_dict = {k: v.detach()
                                      for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict  # cache the unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict

        # Step 3: Store gradients for the generator (if training the generator)
        if train_generator:
            generator_loss, generator_log_dict = self.model.generator_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent,
                initial_latent=image_latent if self.config.i2v else None
            )

            generator_loss_raw = generator_loss.detach()

            if self.gradient_accumulation_steps > 1:
                generator_loss = generator_loss / self.gradient_accumulation_steps

            generator_loss.backward()

            generator_log_dict.update({"generator_loss": generator_loss_raw})

            return generator_log_dict
        else:
            generator_log_dict = {}

        # Step 4: Store gradients for the critic (if training the critic)
        critic_loss, critic_log_dict = self.model.critic_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            initial_latent=image_latent if self.config.i2v else None
        )
        critic_loss_raw = critic_loss.detach()

        if self.gradient_accumulation_steps > 1:
            critic_loss = critic_loss / self.gradient_accumulation_steps

        critic_loss.backward()

        critic_log_dict.update({"critic_loss": critic_loss_raw})

        return critic_log_dict

    def generate_video(self, pipeline, prompts, image=None):
        batch_size = len(prompts)
        if image is not None:
            image = image.squeeze(0).unsqueeze(0).unsqueeze(2).to(device="cuda", dtype=torch.bfloat16)

            # Encode the input image as the first latent
            initial_latent = pipeline.vae.encode_to_latent(image).to(device="cuda", dtype=torch.bfloat16)
            initial_latent = initial_latent.repeat(batch_size, 1, 1, 1, 1)
            sampled_noise = torch.randn(
                [batch_size, self.model.num_training_frames - 1, 16, 60, 104],
                device="cuda",
                dtype=self.dtype
            )
        else:
            initial_latent = None
            sampled_noise = torch.randn(
                [batch_size, self.model.num_training_frames, 16, 60, 104],
                device="cuda",
                dtype=self.dtype
            )

        video, _ = pipeline.inference(
            noise=sampled_noise,
            text_prompts=prompts,
            return_latents=True,
            initial_latent=initial_latent
        )
        current_video = video.permute(0, 1, 3, 4, 2).cpu().numpy() * 255.0
        return current_video

    def train(self):
        start_step = self.step

        while self.step < self.config.full_training_steps:
            # Determine TRAIN_GENERATOR at the optimizer-step level so it stays
            # consistent for the entire accumulation window and gives the correct
            # dfake_gen_update_ratio between effective optimizer steps.
            optimizer_step = self.step // self.gradient_accumulation_steps
            TRAIN_GENERATOR = optimizer_step % self.config.dfake_gen_update_ratio == 0

            # Zero grads at the start of each accumulation window
            if self.step % self.gradient_accumulation_steps == 0:
                self.critic_optimizer.zero_grad(set_to_none=True)
                if TRAIN_GENERATOR:
                    self.generator_optimizer.zero_grad(set_to_none=True)
                self._gen_log_accum = []
                self._critic_log_accum = []

            # Train the generator (accumulate gradients every step of the window)
            if TRAIN_GENERATOR:
                batch = next(self.dataloader)
                extra = self.fwdbwd_one_step(batch, True)
                self._gen_log_accum.append(merge_dict_list([extra]))

            # Train the critic (accumulate gradients every step)
            batch = next(self.dataloader)
            extra = self.fwdbwd_one_step(batch, False)
            self._critic_log_accum.append(merge_dict_list([extra]))

            # Perform optimizer step once at the end of each accumulation window
            if (self.step + 1) % self.gradient_accumulation_steps == 0:
                critic_grad_norm = self.model.fake_score.clip_grad_norm_(
                    self.max_grad_norm_critic)
                self.critic_optimizer.step()
                if TRAIN_GENERATOR:
                    generator_grad_norm = self.model.generator.clip_grad_norm_(
                        self.max_grad_norm_generator)
                    self.generator_optimizer.step()
                    if self.generator_ema is not None:
                        self.generator_ema.update(self.model.generator)

            # Increment the step since we finished gradient update
            self.step += 1

            # Create EMA params (if not already created)
            if (self.step >= self.config.ema_start_step) and \
                    (self.generator_ema is None) and (self.config.ema_weight > 0):
                self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight)

            # Save the model
            if (not self.config.no_save) and (self.step - start_step) > 0 and self.step % self.config.log_iters == 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            # Logging — only at optimizer step boundary
            is_window_end = self.step % self.gradient_accumulation_steps == 0
            if self.is_main_process and is_window_end:
                wandb_loss_dict = {}
                if TRAIN_GENERATOR and self._gen_log_accum:
                    gen_log_dict_avg = merge_dict_list(self._gen_log_accum)
                    gen_log_dict_avg.update({"generator_grad_norm": generator_grad_norm})
                    wandb_loss_dict.update(
                        {
                            "generator_loss": gen_log_dict_avg["generator_loss"].mean().item(),
                            "generator_grad_norm": gen_log_dict_avg["generator_grad_norm"].mean().item(),
                            "dmdtrain_gradient_norm": gen_log_dict_avg["dmdtrain_gradient_norm"].mean().item()
                        }
                    )
                else:
                    gen_log_dict_avg = {}

                critic_log_dict_avg = merge_dict_list(self._critic_log_accum)
                critic_log_dict_avg.update({"critic_grad_norm": critic_grad_norm})

                wandb_loss_dict.update(
                    {
                        "critic_loss": critic_log_dict_avg["critic_loss"].mean().item(),
                        "critic_grad_norm": critic_log_dict_avg["critic_grad_norm"].mean().item()
                    }
                )

                if not self.disable_wandb:
                    wandb.log(wandb_loss_dict, step=self.step)

                # Write to file log every optimizer step
                self._write_log(
                    step=self.step,
                    gen_dict=gen_log_dict_avg,
                    crit_dict=critic_log_dict_avg,
                )

            if self.step % self.config.gc_interval == 0:
                if dist.get_rank() == 0:
                    logging.info("DistGarbageCollector: Running GC.")
                gc.collect()
                torch.cuda.empty_cache()

            if self.is_main_process and is_window_end:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    if not self.disable_wandb:
                        wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                    self.previous_time = current_time

        # Training complete — flush log and generate plots
        if self.is_main_process:
            if self._log_file is not None:
                self._log_file.close()
            with open(os.path.join(self.output_path, "train_log.txt"), "a") as f:
                f.write(f"\n# Training finished at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, "
                        f"total steps={self.step}\n")
            self._plot_metrics()
