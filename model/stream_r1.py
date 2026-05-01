from collections import deque

from pipeline import RewardForcingTrainingPipeline
import torch.nn.functional as F
from typing import Optional, Tuple
import torch

from model.base import RewardForcingModel

class StreamR1(RewardForcingModel):
    def __init__(self, args, device):
        """
        Stream-R1: Reliability-Complexity Aware Reward Distillation.

        Composes the standard DMD generator loss with three reward-guided
        components, all driven by a single pretrained video reward model:
          - Inter-Reliability weighting  (per-rollout exp(beta * reward) multiplier)
          - Intra-Complexity weighting   (per-pixel reward-gradient saliency,
                                          factored into spatial and temporal weights)
          - Adaptive Reward Balancing    (BalancedOverall mode: penalty on the
                                          std of per-axis VQ/MQ/TA improvement)

        Computes the rewarded generator and fake-score losses in the forward pass.
        """
        super().__init__(args, device)
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.same_step_across_blocks = getattr(args, "same_step_across_blocks", True)
        self.num_training_frames = getattr(args, "num_training_frames", 21)

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        self.independent_first_frame = getattr(args, "independent_first_frame", False)
        if self.independent_first_frame:
            self.generator.model.independent_first_frame = True
        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()
            self.fake_score.enable_gradient_checkpointing()

        # this will be init later with fsdp-wrapped modules
        self.inference_pipeline: RewardForcingTrainingPipeline = None

        # Step 2: Initialize all dmd hyperparameters
        self.num_train_timestep = args.num_train_timestep
        self.min_step = int(0.02 * self.num_train_timestep)
        self.max_step = int(0.98 * self.num_train_timestep)
        if hasattr(args, "real_guidance_scale"):
            self.real_guidance_scale = args.real_guidance_scale
            self.fake_guidance_scale = args.fake_guidance_scale
        else:
            self.real_guidance_scale = args.guidance_scale
            self.fake_guidance_scale = 0.0
        self.timestep_shift = getattr(args, "timestep_shift", 1.0)
        self.ts_schedule = getattr(args, "ts_schedule", True)
        self.ts_schedule_max = getattr(args, "ts_schedule_max", False)
        self.min_score_timestep = getattr(args, "min_score_timestep", 0)
        # reward_mode controls which reward signal is used as loss weight.
        # Supported values: "MQ" (default), "VQ", "TA", "Overall", "BalancedOverall"
        self.reward_mode = getattr(args, "reward_mode", "MQ")

        # ---- BalancedOverall: improvement-balance penalty ----
        # reward_dims: which dimensions to include in BalancedOverall (default all three)
        self.reward_dims = list(getattr(args, "reward_dims", ["VQ", "MQ", "TA"]))
        # lambda for penalising std of per-dim improvement deltas (0 = disabled)
        self.reward_balance_lambda = getattr(args, "reward_balance_lambda", 0.5)
        # sliding window length (in training steps) used to measure improvement
        self.reward_balance_window = getattr(args, "reward_balance_window", 50)
        # number of steps before penalty kicks in (history warm-up)
        self.reward_balance_warmup = getattr(args, "reward_balance_warmup", 50)
        if self.reward_mode == "BalancedOverall":
            self._reward_history = {d: deque(maxlen=self.reward_balance_window)
                                    for d in self.reward_dims}
            self._balance_step = 0

        # ---- Spatial reward localization ----
        self.spatial_reward = getattr(args, "spatial_reward", False)
        # "gradient" (Grad-CAM style, more accurate, uses grad checkpointing) or
        # "attention" (extract ViT attention, zero extra cost but less precise)
        self.spatial_reward_method = getattr(args, "spatial_reward_method", "gradient")
        # Which metrics to compute spatial saliency for
        self.spatial_reward_metrics = list(getattr(args, "spatial_reward_metrics", ["MQ", "VQ", "TA"]))
        # Controls how strongly the spatial mask concentrates loss on salient regions.
        # 0 = uniform (no spatial effect), higher = more concentrated.
        self.spatial_reward_strength = getattr(args, "spatial_reward_strength", 1.0)
        # Floor value to prevent any region from receiving zero gradient
        self.spatial_reward_min_weight = getattr(args, "spatial_reward_min_weight", 0.5)
        # How to combine per-metric saliency maps: "adaptive", "mean", "max"
        # adaptive: weight each metric's map inversely proportional to its reward score
        self.spatial_reward_combination = getattr(args, "spatial_reward_combination", "adaptive")
        # Temperature for adaptive softmax weighting (lower = more peaky)
        self.spatial_reward_temperature = getattr(args, "spatial_reward_temperature", 1.0)
        # When True, compute saliency via gradient w.r.t. *pixels* (through ViT)
        # instead of w.r.t. video_embeds (after ViT).  Produces higher-resolution
        # saliency maps but costs extra VRAM for ViT activations.
        self.spatial_reward_pixel_grad = getattr(args, "spatial_reward_pixel_grad", False)

        # ---- Temporal saliency weighting (data-driven, from gradient saliency) ----
        # When True, decompose the [F, H, W] saliency into separate temporal and
        # spatial signals, normalize each independently, and recombine.
        # This preserves temporal contrast that global normalization would wash out.
        self.temporal_saliency_weighting = getattr(args, "temporal_saliency_weighting", False)
        self.temporal_saliency_strength = getattr(args, "temporal_saliency_strength", 1.0)
        self.temporal_saliency_min_weight = getattr(args, "temporal_saliency_min_weight", 0.3)

        if self.temporal_saliency_weighting and not self.spatial_reward:
            raise ValueError(
                "temporal_saliency_weighting requires spatial_reward=True "
                "since it derives temporal weights from the spatial saliency map."
            )

        if getattr(self.scheduler, "alphas_cumprod", None) is not None:
            self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        else:
            self.scheduler.alphas_cumprod = None

    def _compute_kl_grad(
        self, noisy_image_or_video: torch.Tensor,
        estimated_clean_image_or_video: torch.Tensor,
        timestep: torch.Tensor,
        conditional_dict: dict, unconditional_dict: dict,
        normalization: bool = True
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the KL grad (eq 7 in https://arxiv.org/abs/2311.18828).
        Input:
            - noisy_image_or_video: a tensor with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - estimated_clean_image_or_video: a tensor with shape [B, F, C, H, W] representing the estimated clean image or video.
            - timestep: a tensor with shape [B, F] containing the randomly generated timestep.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - normalization: a boolean indicating whether to normalize the gradient.
        Output:
            - kl_grad: a tensor representing the KL grad.
            - kl_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        # Step 1: Compute the fake score
        _, pred_fake_image_cond = self.fake_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=conditional_dict,
            timestep=timestep
        )

        if self.fake_guidance_scale != 0.0:
            _, pred_fake_image_uncond = self.fake_score(
                noisy_image_or_video=noisy_image_or_video,
                conditional_dict=unconditional_dict,
                timestep=timestep
            )
            pred_fake_image = pred_fake_image_cond + (
                pred_fake_image_cond - pred_fake_image_uncond
            ) * self.fake_guidance_scale
        else:
            pred_fake_image = pred_fake_image_cond

        _, pred_real_image_cond = self.real_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=conditional_dict,
            timestep=timestep
        )

        _, pred_real_image_uncond = self.real_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=unconditional_dict,
            timestep=timestep
        )

        pred_real_image = pred_real_image_cond + (
            pred_real_image_cond - pred_real_image_uncond
        ) * self.real_guidance_scale

        grad = (pred_fake_image - pred_real_image)

        if normalization:
            p_real = (estimated_clean_image_or_video - pred_real_image)
            normalizer = torch.abs(p_real).mean(dim=[1, 2, 3, 4], keepdim=True)
            grad = grad / normalizer
        grad = torch.nan_to_num(grad)

        return grad, {
            "dmdtrain_gradient_norm": torch.mean(torch.abs(grad)).detach(),
            "timestep": timestep.detach()
        }

    def compute_rewarded_distribution_matching_loss(
        self,
        image_or_video: torch.Tensor,
        pixels: torch.Tensor,
        text_prompts: list,
        conditional_dict: dict,
        unconditional_dict: dict,
        gradient_mask: Optional[torch.Tensor] = None,
        denoised_timestep_from: int = 0,
        denoised_timestep_to: int = 0,
        beta: float = 1.0
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the DMD loss (eq 7 in https://arxiv.org/abs/2311.18828).
        Input:
            - image_or_video: a tensor with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - gradient_mask: a boolean tensor with the same shape as image_or_video indicating which pixels to compute loss .
        Output:
            - dmd_loss: a scalar tensor representing the DMD loss.
            - dmd_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        original_latent = image_or_video

        batch_size, num_frame = image_or_video.shape[:2]

        # pixels are in [-1, 1] from VAE decode; convert to [0, 1] float for the reward processor
        videos = ((pixels + 1.0) / 2.0).clamp(0, 1)

        # --- Global reward (existing logic, always computed) ---
        reward = self.inferencer.reward_from_frames(
            [videos[0]],
            [text_prompts[0]],
            use_norm=True,
        )

        with torch.no_grad():
            min_timestep = denoised_timestep_to if self.ts_schedule and denoised_timestep_to is not None else self.min_score_timestep
            max_timestep = denoised_timestep_from if self.ts_schedule_max and denoised_timestep_from is not None else self.num_train_timestep
            timestep = self._get_timestep(
                min_timestep,
                max_timestep,
                batch_size,
                num_frame,
                self.num_frame_per_block,
                uniform_timestep=True
            )

            if self.timestep_shift > 1:
                timestep = self.timestep_shift * \
                    (timestep / 1000) / \
                    (1 + (self.timestep_shift - 1) * (timestep / 1000)) * 1000
            timestep = timestep.clamp(self.min_step, self.max_step)

            noise = torch.randn_like(image_or_video)
            noisy_latent = self.scheduler.add_noise(
                image_or_video.flatten(0, 1),
                noise.flatten(0, 1),
                timestep.flatten(0, 1)
            ).detach().unflatten(0, (batch_size, num_frame))

            grad, rl_dmd_log_dict = self._compute_kl_grad(
                noisy_image_or_video=noisy_latent,
                estimated_clean_image_or_video=original_latent,
                timestep=timestep,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict
            )

        # --- Select base reward signal from global score ---
        combined_reward = self._select_reward(reward)

        # --- BalancedOverall improvement-balance penalty ---
        balance_penalty = torch.zeros(1, device=combined_reward.device, dtype=combined_reward.dtype)
        if self.reward_mode == "BalancedOverall":
            combined_reward, balance_penalty = self._apply_balance_penalty(
                combined_reward, reward)

        # --- Spatial reward localization ---
        spatial_log = {}
        if self.spatial_reward:
            latent_shape = (original_latent.shape[1], original_latent.shape[3], original_latent.shape[4])  # (F, H, W)
            spatial_weight, component_stats = self._compute_spatial_reward_mask(
                videos, text_prompts, reward, latent_shape)  # [1, F, 1, H, W]

            # Spatially-weighted element-wise MSE loss
            target = (original_latent.double() - grad.double()).detach()
            element_loss = (original_latent.double() - target) ** 2  # [B, F, C, H, W]
            spatial_weight_d = spatial_weight.double()

            if gradient_mask is not None:
                weighted_loss = (spatial_weight_d * element_loss)[gradient_mask]
            else:
                weighted_loss = spatial_weight_d * element_loss

            rl_dmd_loss = 0.5 * torch.exp(beta * combined_reward) * weighted_loss.mean()

            spatial_log.update(component_stats)
        else:
            if gradient_mask is not None:
                rl_dmd_loss = 0.5 * torch.exp(beta * combined_reward) * F.mse_loss(original_latent.double(
                )[gradient_mask], (original_latent.double() - grad.double()).detach()[gradient_mask], reduction="mean")
            else:
                rl_dmd_loss = 0.5 * torch.exp(beta * combined_reward) * F.mse_loss(original_latent.double(
                ), (original_latent.double() - grad.double()).detach(), reduction="mean")

        rl_dmd_log_dict["reward_vq"] = reward['VQ'].detach().float()
        rl_dmd_log_dict["reward_mq"] = reward['MQ'].detach().float()
        rl_dmd_log_dict["reward_ta"] = reward['TA'].detach().float()
        rl_dmd_log_dict["reward_weight"] = torch.exp(beta * combined_reward).detach().float()
        rl_dmd_log_dict["reward_balance_penalty"] = balance_penalty.detach().float()
        rl_dmd_log_dict.update(spatial_log)
        return rl_dmd_loss, rl_dmd_log_dict

    # ------------------------------------------------------------------
    # Helper: spatial reward mask computation
    # ------------------------------------------------------------------
    def _compute_spatial_reward_mask(self, videos, text_prompts, reward_dict, latent_shape):
        """Compute a spatial weight mask for the DMD loss.

        Args:
            videos: pixel tensor [B, T, C, H, W] in [0, 1]
            text_prompts: list of strings
            reward_dict: dict with per-metric reward scalars (VQ, MQ, TA)
            latent_shape: (F, H, W) of the latent tensor

        Returns:
            spatial_weight: [1, F, 1, H, W] tensor, mean-normalized to 1.0
            component_stats: dict with separate spatial/temporal statistics
        """
        device = videos.device
        dtype = videos.dtype

        if self.spatial_reward_method == "gradient" and self.spatial_reward_pixel_grad:
            # Pixel-level gradient saliency (through ViT, higher resolution, extra VRAM)
            saliency_maps = self.inferencer.compute_pixel_spatial_saliency(
                [videos[0]], [text_prompts[0]],
                metrics=tuple(self.spatial_reward_metrics),
                latent_shape=latent_shape,
            )
        elif self.spatial_reward_method == "gradient":
            # Patch-level gradient saliency (default, after ViT)
            saliency_maps = self.inferencer.compute_spatial_saliency(
                [videos[0]], [text_prompts[0]],
                metrics=tuple(self.spatial_reward_metrics),
                latent_shape=latent_shape,
            )
        else:  # "attention"
            attn_map = self.inferencer.extract_attention_spatial_map(
                [videos[0]], [text_prompts[0]],
                latent_shape=latent_shape,
            )
            # Use the same attention map for all metrics
            saliency_maps = {m: attn_map for m in self.spatial_reward_metrics}

        # Combine per-metric saliency maps
        if self.spatial_reward_combination == "adaptive" and len(self.spatial_reward_metrics) > 1:
            # Lower reward → higher weight for that metric's saliency
            reward_vals = torch.tensor(
                [reward_dict[m].item() for m in self.spatial_reward_metrics],
                device=device, dtype=torch.float32
            )
            weights = torch.softmax(-reward_vals / self.spatial_reward_temperature, dim=0)
            combined = sum(
                weights[i].item() * saliency_maps[m].to(device).float()
                for i, m in enumerate(self.spatial_reward_metrics)
            )
        elif self.spatial_reward_combination == "max":
            stacked = torch.stack([saliency_maps[m].to(device).float()
                                   for m in self.spatial_reward_metrics])
            combined = stacked.max(dim=0).values
        else:  # "mean"
            combined = sum(saliency_maps[m].to(device).float()
                           for m in self.spatial_reward_metrics) / len(self.spatial_reward_metrics)

        if self.temporal_saliency_weighting:
            # === Factored normalization: separate temporal and spatial ===
            F_dim = combined.shape[0]

            # 1. Extract temporal profile: per-frame mean saliency magnitude
            temporal_profile = combined.mean(dim=(1, 2))  # [F]

            # 2. Normalize temporal profile to [0, 1]
            t_min = temporal_profile.min()
            t_max = temporal_profile.max()
            if t_max - t_min > 1e-8:
                temporal_norm = (temporal_profile - t_min) / (t_max - t_min)
            else:
                temporal_norm = torch.ones_like(temporal_profile)

            # 3. Apply temporal strength and floor, then mean-normalize
            temporal_weight = (
                (1.0 - self.temporal_saliency_strength)
                + self.temporal_saliency_strength * temporal_norm
            )
            temporal_weight = temporal_weight.clamp(min=self.temporal_saliency_min_weight)
            temporal_weight = temporal_weight / temporal_weight.mean()

            # 4. Per-frame spatial normalization: each frame gets full [0, 1] range
            spatial_normed = torch.empty_like(combined)
            for f in range(F_dim):
                frame = combined[f]  # [H, W]
                f_min = frame.min()
                f_max = frame.max()
                if f_max - f_min > 1e-8:
                    spatial_normed[f] = (frame - f_min) / (f_max - f_min)
                else:
                    spatial_normed[f] = torch.ones_like(frame)

            # 5. Apply spatial strength and floor
            spatial_weight = (
                (1.0 - self.spatial_reward_strength)
                + self.spatial_reward_strength * spatial_normed
            )
            spatial_weight = spatial_weight.clamp(min=self.spatial_reward_min_weight)
            # Per-frame mean-normalize spatial weights
            for f in range(F_dim):
                frame_mean = spatial_weight[f].mean()
                if frame_mean > 1e-8:
                    spatial_weight[f] = spatial_weight[f] / frame_mean

            # Record component-level stats before combining
            component_stats = {
                "spatial_weight_min": spatial_weight.min().detach().float(),
                "spatial_weight_max": spatial_weight.max().detach().float(),
                "spatial_weight_std": spatial_weight.std().detach().float(),
                "temporal_sal_weight_min": temporal_weight.min().detach().float(),
                "temporal_sal_weight_max": temporal_weight.max().detach().float(),
                "temporal_sal_weight_std": temporal_weight.std().detach().float(),
                "temporal_sal_weight_per_frame": temporal_weight.detach().cpu().tolist(),
            }

            # 6. Combine: temporal[F] × spatial[F, H, W]
            final_weight = temporal_weight.unsqueeze(1).unsqueeze(2) * spatial_weight

            # 7. Final mean-normalization for overall scale preservation
            final_weight = final_weight / final_weight.mean()

            spatial_weight = final_weight.unsqueeze(0).unsqueeze(2).to(dtype)  # [1, F, 1, H, W]
        else:
            # === Original global normalization (unchanged) ===
            sal_min = combined.min()
            sal_max = combined.max()
            if sal_max - sal_min > 1e-8:
                combined = (combined - sal_min) / (sal_max - sal_min)
            else:
                combined = torch.ones_like(combined)

            spatial_weight = (1.0 - self.spatial_reward_strength) + self.spatial_reward_strength * combined
            spatial_weight = spatial_weight.clamp(min=self.spatial_reward_min_weight)
            spatial_weight = spatial_weight / spatial_weight.mean()
            spatial_weight = spatial_weight.unsqueeze(0).unsqueeze(2).to(dtype)
            component_stats = {
                "spatial_weight_min": spatial_weight.min().detach().float(),
                "spatial_weight_max": spatial_weight.max().detach().float(),
                "spatial_weight_std": spatial_weight.std().detach().float(),
            }

        return spatial_weight, component_stats

    # ------------------------------------------------------------------
    # Helper: select scalar reward from a reward dict based on reward_mode
    # ------------------------------------------------------------------
    def _select_reward(self, reward_dict):
        mode = self.reward_mode
        if mode == "VQ":
            return reward_dict['VQ']
        elif mode == "TA":
            return reward_dict['TA']
        elif mode in ("Overall", "BalancedOverall"):
            return sum(reward_dict[d] for d in self.reward_dims) / len(self.reward_dims)
        else:  # "MQ" (default)
            return reward_dict['MQ']

    # ------------------------------------------------------------------
    # Helper: BalancedOverall improvement-balance penalty
    # ------------------------------------------------------------------
    def _apply_balance_penalty(self, combined_reward, reward_dict):
        """
        Track a sliding window of per-dim scores.
        After warmup, penalise the std of per-dim improvement deltas so that
        all selected dimensions improve at roughly the same rate.
        """
        self._balance_step += 1

        # Append current scores to history
        for d in self.reward_dims:
            self._reward_history[d].append(reward_dict[d].item())

        penalty = torch.zeros((), device=combined_reward.device, dtype=combined_reward.dtype)

        if (self._balance_step >= self.reward_balance_warmup
                and all(len(self._reward_history[d]) >= 4 for d in self.reward_dims)):
            # Split window into early half (baseline) and recent half
            deltas = []
            for d in self.reward_dims:
                hist = list(self._reward_history[d])
                half = max(1, len(hist) // 2)
                baseline = sum(hist[:half]) / half
                recent = sum(hist[half:]) / max(1, len(hist) - half)
                deltas.append(recent - baseline)

            if len(deltas) > 1:
                deltas_t = torch.tensor(deltas, device=combined_reward.device,
                                        dtype=combined_reward.dtype)
                penalty = deltas_t.std()
                combined_reward = combined_reward - self.reward_balance_lambda * penalty

        return combined_reward, penalty

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        text_prompts: list,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None,
        beta: float = 1.0
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noise and compute the DMD loss.
        The noisy input to the generator is backward simulated.
        This removes the need of any datasets during distillation.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
        Output:
            - loss: a scalar tensor representing the generator loss.
            - generator_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        # Step 1: Unroll generator to obtain fake videos
        pred_image, gradient_mask, denoised_timestep_from, denoised_timestep_to, pixels = self._run_generator(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            initial_latent=initial_latent
        )

        # Step 2: Compute the DMD loss
        rl_dmd_loss, rl_dmd_log_dict = self.compute_rewarded_distribution_matching_loss(   
            image_or_video=pred_image,
            pixels = pixels,
            text_prompts=text_prompts,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            gradient_mask=gradient_mask,
            denoised_timestep_from=denoised_timestep_from,
            denoised_timestep_to=denoised_timestep_to,
            beta = beta
        )

        return rl_dmd_loss, rl_dmd_log_dict

    def critic_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noise and train the critic with generated samples.
        The noisy input to the generator is backward simulated.
        This removes the need of any datasets during distillation.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
        Output:
            - loss: a scalar tensor representing the generator loss.
            - critic_log_dict: a dictionary containing the intermediate tensors for logging.
        """

        # Step 1: Run generator on backward simulated noisy input
        with torch.no_grad():
            generated_image, _, denoised_timestep_from, denoised_timestep_to, _ = self._run_generator(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                initial_latent=initial_latent
            )

        # Step 2: Compute the fake prediction
        min_timestep = denoised_timestep_to if self.ts_schedule and denoised_timestep_to is not None else self.min_score_timestep
        max_timestep = denoised_timestep_from if self.ts_schedule_max and denoised_timestep_from is not None else self.num_train_timestep
        critic_timestep = self._get_timestep(
            min_timestep,
            max_timestep,
            image_or_video_shape[0],
            image_or_video_shape[1],
            self.num_frame_per_block,
            uniform_timestep=True
        )

        if self.timestep_shift > 1:
            critic_timestep = self.timestep_shift * \
                (critic_timestep / 1000) / (1 + (self.timestep_shift - 1) * (critic_timestep / 1000)) * 1000

        critic_timestep = critic_timestep.clamp(self.min_step, self.max_step)

        critic_noise = torch.randn_like(generated_image)
        noisy_generated_image = self.scheduler.add_noise(
            generated_image.flatten(0, 1),
            critic_noise.flatten(0, 1),
            critic_timestep.flatten(0, 1)
        ).unflatten(0, image_or_video_shape[:2])

        _, pred_fake_image = self.fake_score(
            noisy_image_or_video=noisy_generated_image,
            conditional_dict=conditional_dict,
            timestep=critic_timestep
        )

        # Step 3: Compute the denoising loss for the fake critic
        if self.args.denoising_loss_type == "flow":
            from utils.wan_wrapper import WanDiffusionWrapper
            flow_pred = WanDiffusionWrapper._convert_x0_to_flow_pred(
                scheduler=self.scheduler,
                x0_pred=pred_fake_image.flatten(0, 1),
                xt=noisy_generated_image.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1)
            )
            pred_fake_noise = None
        else:
            flow_pred = None
            pred_fake_noise = self.scheduler.convert_x0_to_noise(
                x0=pred_fake_image.flatten(0, 1),
                xt=noisy_generated_image.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1)
            ).unflatten(0, image_or_video_shape[:2])

        denoising_loss = self.denoising_loss_func(
            x=generated_image.flatten(0, 1),
            x_pred=pred_fake_image.flatten(0, 1),
            noise=critic_noise.flatten(0, 1),
            noise_pred=pred_fake_noise,
            alphas_cumprod=self.scheduler.alphas_cumprod,
            timestep=critic_timestep.flatten(0, 1),
            flow_pred=flow_pred
        )

        # Step 5: Debugging Log
        critic_log_dict = {
            "critic_timestep": critic_timestep.detach()
        }

        return denoising_loss, critic_log_dict
