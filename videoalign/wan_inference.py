import ast
import json
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pdb
from collections.abc import Mapping
import pandas as pd

import torch
from videoalign.vision_process import process_vision_info, process_wanvideo_tensor

from videoalign.data import DataConfig
from videoalign.utils import ModelConfig, PEFTLoraConfig, TrainingConfig
from videoalign.utils import load_model_from_checkpoint
from videoalign.train_reward import create_model_and_processor
from videoalign.prompt_template import build_prompt

import numpy as np
from PIL import Image
import imageio

def load_configs_from_json(config_path):
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    # del config_dict["training_args"]["_n_gpu"]
    del config_dict["data_config"]["meta_data"]
    del config_dict["data_config"]["data_dir"]

    return config_dict["data_config"], None, config_dict["model_config"], config_dict["peft_lora_config"], \
           config_dict["inference_config"] if "inference_config" in config_dict else None

class VideoVLMRewardInference():
    def __init__(self, load_from_pretrained, load_from_pretrained_step=-1, device='cuda', dtype=torch.bfloat16):
        config_path = os.path.join(load_from_pretrained, "model_config.json")
        data_config, _, model_config, peft_lora_config, inference_config = load_configs_from_json(config_path)
        data_config = DataConfig(**data_config)
        model_config = ModelConfig(**model_config)
        peft_lora_config = PEFTLoraConfig(**peft_lora_config)

        training_args = TrainingConfig(
            load_from_pretrained=load_from_pretrained,
            load_from_pretrained_step=load_from_pretrained_step,
            gradient_checkpointing=False,
            disable_flash_attn2=False,
            bf16=True if dtype == torch.bfloat16 else False,
            fp16=True if dtype == torch.float16 else False,
            output_dir="",
        )
        
        model, processor, peft_config = create_model_and_processor(
            model_config=model_config,
            peft_lora_config=peft_lora_config,
            training_args=training_args,
        )

        self.device = device

        model, checkpoint_step = load_model_from_checkpoint(model, load_from_pretrained, load_from_pretrained_step)
        model.eval()

        self.model = model
        self.processor = processor

        self.model.to(self.device)

        self.data_config = data_config

        self.inference_config = inference_config

    def _norm(self, reward):
        if self.inference_config is None:
            return reward
        else:
            reward['VQ'] = (reward['VQ'] - self.inference_config['VQ_mean']) / self.inference_config['VQ_std']
            reward['MQ'] = (reward['MQ'] - self.inference_config['MQ_mean']) / self.inference_config['MQ_std']
            reward['TA'] = (reward['TA'] - self.inference_config['TA_mean']) / self.inference_config['TA_std']
            return reward

    def _pad_sequence(self, sequences, attention_mask, max_len, padding_side='right'):
        """
        Pad the sequences to the maximum length.
        """
        assert padding_side in ['right', 'left']
        if sequences.shape[1] >= max_len:
            return sequences, attention_mask
        
        pad_len = max_len - sequences.shape[1]
        padding = (0, pad_len) if padding_side == 'right' else (pad_len, 0)

        sequences_padded = torch.nn.functional.pad(sequences, padding, 'constant', self.processor.tokenizer.pad_token_id)
        attention_mask_padded = torch.nn.functional.pad(attention_mask, padding, 'constant', 0)

        return sequences_padded, attention_mask_padded
    
    def _prepare_input(self, data):
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = {"device": self.device}
            ## TODO: Maybe need to add dtype
            # if self.is_deepspeed_enabled and (torch.is_floating_point(data) or torch.is_complex(data)):
            #     # NLP models inputs are int/uint and those get adjusted to the right dtype of the
            #     # embedding. Other models such as wav2vec2's inputs are already float and thus
            #     # may need special handling to match the dtypes of the model
            #     kwargs.update({"dtype": self.accelerator.state.deepspeed_plugin.hf_ds_config.dtype()})
            return data.to(**kwargs)
        return data
    
    def _prepare_inputs(self, inputs):
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        inputs = self._prepare_input(inputs)
        if len(inputs) == 0:
            raise ValueError
        return inputs

    def prepare_batch_from_frames(self, video_tensors, prompts, fps=None, num_frames=None, max_pixels=None,):
        """
        Args:
            video_tensors: List[torch.Tensor], shape [T, C, H, W]
            prompts: List[str]
        """
        fps = self.data_config.fps if fps is None else fps
        num_frames = self.data_config.num_frames if num_frames is None else num_frames
        max_pixels = self.data_config.max_frame_pixels if max_pixels is None else max_pixels

        chat_data = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": "file://dummy_path"},  
                        {"type": "text", "text": build_prompt(prompt, self.data_config.eval_dim, self.data_config.prompt_template_type)},
                    ],
                }
            ] for prompt in prompts
        ]

        processed_videos = []
        for tensor in video_tensors:
            processed = process_wanvideo_tensor(tensor)
            processed_videos.append(processed)
        video_tensors = processed_videos
        batch = self.processor(
            text=self.processor.apply_chat_template(chat_data, tokenize=False, add_generation_prompt=True),
            images=None,
            videos=video_tensors,
            padding=True,
            return_tensors="pt",
            videos_kwargs={"do_rescale": False},  # input is already [0, 1] float
        )

        return self._prepare_inputs(batch)


    def reward_from_frames(self, video_tensors, prompts, use_norm=True):
        batch = self.prepare_batch_from_frames(video_tensors, prompts)
        rewards = self.model(**batch, return_dict=True)["logits"]  # [B, 3]
        
        if use_norm:
            vq = (rewards[:, 0] - self.inference_config['VQ_mean']) / self.inference_config['VQ_std']
            mq = (rewards[:, 1] - self.inference_config['MQ_mean']) / self.inference_config['MQ_std']
            ta = (rewards[:, 2] - self.inference_config['TA_mean']) / self.inference_config['TA_std']
        else:
            vq = rewards[:, 0]
            mq = rewards[:, 1]
            ta = rewards[:, 2]
        
        overall = (1 * mq + 1 * vq + 1 * ta)/3
        
        return {
            'VQ': vq,
            'MQ': mq,
            'TA': ta,
            'Overall': overall
        }
    
    def compute_spatial_saliency(self, video_tensors, prompts, metrics=("MQ", "VQ", "TA"),
                                 latent_shape=None):
        """Compute per-metric spatial saliency maps via gradient backprop through reward model.

        Args:
            video_tensors: List[torch.Tensor] of shape [T, C, H, W] in [0, 1]
            prompts: List[str]
            metrics: tuple of metric names to compute saliency for
            latent_shape: optional (F, H, W) of the latent space for interpolation

        Returns:
            dict mapping metric name -> spatial saliency tensor.
            If latent_shape is given, shape is [F, H, W]; otherwise [T_patch, H_patch, W_patch].
        """
        import torch.nn.functional as nnf

        metric_to_idx = {"VQ": 0, "MQ": 1, "TA": 2}
        batch = self.prepare_batch_from_frames(video_tensors, prompts)

        # Enable gradient checkpointing on LLM backbone to save VRAM
        had_grad_ckpt = getattr(self.model.model, 'gradient_checkpointing', False)
        self.model.model.gradient_checkpointing_enable()

        try:
            # Forward with video_embeds exposed for gradient
            result = self.model.forward_with_embeds(**batch, return_dict=True)
            logits = result["logits"]           # [B, output_dim]
            video_embeds = result["video_embeds"]   # [num_video_tokens, D]
            grid_thw = result["video_grid_thw"]     # [1, 3] or [num_videos, 3]

            t_p, h_p, w_p = grid_thw[0].tolist()
            t_p, h_p, w_p = int(t_p), int(h_p), int(w_p)

            # Qwen2-VL's visual merger pools 2x2 spatial patches into one token,
            # so the actual token grid is (t, h/2, w/2).
            spatial_merge_size = getattr(self.model.config, 'spatial_merge_size', 2)
            h_p = h_p // spatial_merge_size
            w_p = w_p // spatial_merge_size

            saliency_maps = {}
            num_metrics = len(metrics)
            for idx_m, metric in enumerate(metrics):
                score = logits[0, metric_to_idx[metric]]
                grad = torch.autograd.grad(
                    score, video_embeds,
                    retain_graph=(idx_m < num_metrics - 1),
                )[0]  # same shape as video_embeds: [num_video_tokens, D]

                # L2 norm across hidden dim → per-token saliency
                sal = grad.norm(dim=-1)  # [num_video_tokens]

                # Reshape to spatial grid
                sal = sal.view(t_p, h_p, w_p)

                # Interpolate to latent space if requested
                if latent_shape is not None:
                    f_lat, h_lat, w_lat = latent_shape
                    sal = sal.unsqueeze(0).unsqueeze(0).float()  # [1, 1, T, H, W]
                    sal = nnf.interpolate(sal, size=(f_lat, h_lat, w_lat), mode='trilinear', align_corners=False)
                    sal = sal.squeeze(0).squeeze(0)  # [F, H, W]

                saliency_maps[metric] = sal.detach()

        finally:
            # Restore original gradient checkpointing state
            if not had_grad_ckpt:
                self.model.model.gradient_checkpointing_disable()

        return saliency_maps

    def compute_pixel_spatial_saliency(self, video_tensors, prompts, metrics=("MQ", "VQ", "TA"),
                                       latent_shape=None):
        """Compute per-metric spatial saliency via gradient w.r.t. *pixels* (through ViT).

        Unlike compute_spatial_saliency which computes grad w.r.t. video_embeds (patch-level),
        this method lets gradients flow through the visual encoder, producing pixel-resolution
        saliency maps.  The trade-off is extra VRAM for ViT activations.

        Args:
            video_tensors: List[torch.Tensor] of shape [T, C, H, W] in [0, 1]
            prompts: List[str]
            metrics: tuple of metric names to compute saliency for
            latent_shape: optional (F, H, W) of the latent space for interpolation

        Returns:
            dict mapping metric name -> spatial saliency tensor.
            If latent_shape is given, shape is [F, H, W]; otherwise [T_pixel, H_pixel, W_pixel].
        """
        import torch.nn.functional as nnf

        metric_to_idx = {"VQ": 0, "MQ": 1, "TA": 2}
        batch = self.prepare_batch_from_frames(video_tensors, prompts)

        # Enable gradient checkpointing on LLM backbone to save VRAM
        had_grad_ckpt = getattr(self.model.model, 'gradient_checkpointing', False)
        self.model.model.gradient_checkpointing_enable()

        # Also enable gradient checkpointing on the visual encoder (ViT) —
        # without this, all ViT activations are kept for backward, causing OOM.
        # Qwen2VisionTransformerPretrainedModel does NOT support the standard
        # gradient_checkpointing_enable(), so we manually wrap each block.
        visual = self.model.visual
        from torch.utils.checkpoint import checkpoint as ckpt_fn
        _original_block_forwards = []
        for blk in visual.blocks:
            _original_block_forwards.append(blk.forward)
            def _make_ckpt_fwd(fn):
                def _ckpt_fwd(*args, **kwargs):
                    return ckpt_fn(fn, *args, use_reentrant=False, **kwargs)
                return _ckpt_fwd
            blk.forward = _make_ckpt_fwd(blk.forward)

        try:
            # Forward with pixel_grad=True: grad target is pixel_values_videos
            result = self.model.forward_with_embeds(**batch, return_dict=True, pixel_grad=True)
            logits = result["logits"]                       # [B, output_dim]
            pixel_input = result["pixel_values_videos"]     # [N_pixels, C] or flattened pixel tensor
            grid_thw = result["video_grid_thw"]             # [1, 3]

            # pixel_input comes from the Qwen2-VL processor: it's a flat tensor of
            # shape [num_patches * temporal_patch * patch_H * patch_W, C_in].
            # We need the original spatial dims from grid_thw (before spatial merging).
            t_p, h_p, w_p = grid_thw[0].tolist()
            t_p, h_p, w_p = int(t_p), int(h_p), int(w_p)

            saliency_maps = {}
            num_metrics = len(metrics)
            for idx_m, metric in enumerate(metrics):
                score = logits[0, metric_to_idx[metric]]
                grad = torch.autograd.grad(
                    score, pixel_input,
                    retain_graph=(idx_m < num_metrics - 1),
                )[0]  # same shape as pixel_input

                # Per-element L2 norm across channel dim → per-pixel saliency
                # pixel_input shape from Qwen2-VL: [num_patches * t_patch * h_patch * w_patch, C]
                # where t_patch=2, h_patch=14, w_patch=14 (default patch sizes)
                sal = grad.norm(dim=-1)  # [N]

                # Reshape to the patch grid.  Each grid cell = temporal_patch_size * patch_size^2 pixels.
                # Total pixels = t_p * h_p * w_p * (temporal_patch * spatial_patch^2)
                # We average-pool within each grid cell to get [t_p, h_p, w_p].
                num_grid = t_p * h_p * w_p
                pixels_per_cell = sal.numel() // num_grid
                sal = sal.view(num_grid, pixels_per_cell).mean(dim=1)  # [num_grid]
                sal = sal.view(t_p, h_p, w_p)

                # Interpolate to latent space if requested
                if latent_shape is not None:
                    f_lat, h_lat, w_lat = latent_shape
                    sal = sal.unsqueeze(0).unsqueeze(0).float()  # [1, 1, T, H, W]
                    sal = nnf.interpolate(sal, size=(f_lat, h_lat, w_lat), mode='trilinear', align_corners=False)
                    sal = sal.squeeze(0).squeeze(0)  # [F, H, W]

                saliency_maps[metric] = sal.detach()

        finally:
            if not had_grad_ckpt:
                self.model.model.gradient_checkpointing_disable()
            # Restore original ViT block forwards
            for blk, orig_fwd in zip(visual.blocks, _original_block_forwards):
                blk.forward = orig_fwd

        return saliency_maps

    def extract_attention_spatial_map(self, video_tensors, prompts, latent_shape=None):
        """Extract spatial attention map from reward model visual encoder (no backward needed).

        Args:
            video_tensors: List[torch.Tensor] of shape [T, C, H, W] in [0, 1]
            prompts: List[str]
            latent_shape: optional (F, H, W) for interpolation

        Returns:
            attn_map: tensor of shape [T_patch, H_patch, W_patch] or [F, H, W] if latent_shape given.
        """
        import torch.nn.functional as nnf

        batch = self.prepare_batch_from_frames(video_tensors, prompts)
        with torch.no_grad():
            result = self.model.forward_with_embeds(**batch, return_dict=True)

        grid_thw = result["video_grid_thw"]  # [1, 3]
        t_p, h_p, w_p = grid_thw[0].tolist()
        t_p, h_p, w_p = int(t_p), int(h_p), int(w_p)

        # Qwen2-VL's spatial merger pools 2x2 spatial patches into one token
        spatial_merge_size = getattr(self.model.config, 'spatial_merge_size', 2)
        h_p = h_p // spatial_merge_size
        w_p = w_p // spatial_merge_size

        if "visual_attn_weights" in result and result["visual_attn_weights"] is not None:
            # attn_weights: [num_heads, num_tokens, num_tokens]
            attn = result["visual_attn_weights"]
            # Average over heads, take the mean attention received by each token
            attn_map = attn.mean(dim=0).mean(dim=0)  # [num_tokens]
        else:
            # Fallback: use hidden state norms from video_embeds as proxy
            video_embeds = result.get("video_embeds")
            if video_embeds is not None:
                attn_map = video_embeds.detach().norm(dim=-1)  # [num_tokens]
            else:
                # No spatial info available, return uniform
                total_tokens = t_p * h_p * w_p
                attn_map = torch.ones(total_tokens, device=self.device)

        attn_map = attn_map.view(t_p, h_p, w_p)

        if latent_shape is not None:
            f_lat, h_lat, w_lat = latent_shape
            attn_map = attn_map.unsqueeze(0).unsqueeze(0).float()
            attn_map = nnf.interpolate(attn_map, size=(f_lat, h_lat, w_lat),
                                       mode='trilinear', align_corners=False)
            attn_map = attn_map.squeeze(0).squeeze(0)

        return attn_map.detach()

    def prepare_batch(self, video_paths, prompts, fps=None, num_frames=None, max_pixels=None,):
        fps = self.data_config.fps if fps is None else fps
        num_frames = self.data_config.num_frames if num_frames is None else num_frames
        max_pixels = self.data_config.max_frame_pixels if max_pixels is None else max_pixels

        if num_frames is None:
            chat_data = [
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video", 
                                "video": f"file://{video_path}", 
                                "max_pixels": max_pixels, 
                                "fps": fps,
                                "sample_type": self.data_config.sample_type,
                            },
                            {"type": "text", "text": build_prompt(prompt, self.data_config.eval_dim, self.data_config.prompt_template_type)},
                        ],
                    },
                ] for video_path, prompt in zip(video_paths, prompts)
            ]
        else:
            chat_data = [
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": f"file://{video_path}", 
                                "max_pixels": max_pixels, 
                                "nframes": num_frames,
                                "sample_type": self.data_config.sample_type,
                            },
                            {"type": "text", "text": build_prompt(prompt, self.data_config.eval_dim, self.data_config.prompt_template_type)},
                        ],
                    },
                ] for video_path, prompt in zip(video_paths, prompts)
            ]
        image_inputs, video_inputs = process_vision_info(chat_data)

        batch = self.processor(
            text=self.processor.apply_chat_template(chat_data, tokenize=False, add_generation_prompt=True),
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            videos_kwargs={"do_rescale": True},
        )
        batch = self._prepare_inputs(batch)
        return batch

    def reward(self, video_paths, prompts, fps=None, num_frames=None, max_pixels=None, use_norm=True):
        """
        Inputs:
            video_paths: List[str], B paths of the videos.
            prompts: List[str], B prompts for the videos.
            eval_dims: List[str], N evaluation dimensions.
            fps: float, sample rate of the videos. If None, use the default value in the config.
            num_frames: int, number of frames of the videos. If None, use the default value in the config.
            max_pixels: int, maximum pixels of the videos. If None, use the default value in the config.
            use_norm: bool, whether to rescale the output rewards
        Outputs:
            Rewards: List[dict], N + 1 rewards of the B videos.
        """
        assert fps is None or num_frames is None, "fps and num_frames cannot be set at the same time."
        
        batch = self.prepare_batch(video_paths, prompts, fps, num_frames, max_pixels)
        rewards = self.model(
            return_dict=True,
            **batch
        )["logits"]

        rewards = [{'VQ': reward[0].item(), 'MQ': reward[1].item(), 'TA': reward[2].item()} for reward in rewards]
        for i in range(len(rewards)):
            if use_norm:
                rewards[i] = self._norm(rewards[i])
            rewards[i]['Overall'] = rewards[i]['VQ'] + rewards[i]['MQ'] + rewards[i]['TA']

        return rewards


if __name__ == "__main__":
    load_from_pretrained = "models/VideoReward"
    device = "cuda:0"
    dtype = torch.bfloat16

    inferencer = VideoVLMRewardInference(load_from_pretrained, device=device, dtype=dtype)

    video_paths = [
        "test/video1.mp4",
        "test/video2.mp4",
        "test/video3.mp4",
    ]

    prompts = ["A stylish woman strolls down a bustling Tokyo street, the warm glow of neon lights and animated city signs casting vibrant reflections. She wears a sleek black leather jacket paired with a flowing red dress and black boots, her black purse slung over her shoulder. Sunglasses perched on her nose and a bold red lipstick add to her confident, casual demeanor. The street is damp and reflective, creating a mirror-like effect that enhances the colorful lights and shadows. Pedestrians move about, adding to the lively atmosphere. The scene is captured in a dynamic medium shot with the woman walking slightly to one side, highlighting her graceful strides.",
                "A stunning mid-afternoon landscape photograph with a low camera angle, showcasing several giant wooly mammoths treading through a snowy meadow. Their long, wooly fur gently billows in the brisk wind as they move, creating a sense of natural movement. Snow-covered trees and dramatic snow-capped mountains loom in the distance, adding to the majestic setting. Wispy clouds and a high sun cast a warm glow over the scene, enhancing the serene and awe-inspiring atmosphere. The depth of field brings out the detailed textures of the mammoths and the snowy environment, capturing every nuance of these prehistoric giants in breathtaking clarity.",
                "A movie trailer in a classic cinematic style, featuring the adventurous journey of a 30-year-old space man wearing a vibrant red wool knitted motorcycle helmet. The scene unfolds against a vast blue sky and a desolate salt desert landscape. Shot on 35mm film, the trailer showcases vivid and rich colors, capturing the hero as he navigates through the harsh terrain with determination. His helmet glints under the sun, adding to the dramatic effect. The background is a mix of sweeping desert vistas and distant horizons, with the occasional shimmer of light reflecting off the salt flats. A dynamic medium shot with a sweeping overhead angle, emphasizing the hero's resilience and the vastness of his adventure."]
    with torch.no_grad():
        rewards = inferencer.reward(video_paths, prompts, use_norm=True)
        print(rewards)

    def load_video_frames(video_path, num_frames=81):
        try:
            reader = imageio.get_reader(video_path)
            frames = []
            for i, frame in enumerate(reader):
                if i >= num_frames:
                    break
                img = Image.fromarray(frame)
                frames.append(np.array(img))
            
            # 转换为张量 [T, H, W, C] -> [T, C, H, W]
            tensor = torch.tensor(np.stack(frames)).permute(0, 3, 1, 2)
            print(tensor.shape)
            return tensor
        except Exception as e:
            print(f"fail: {video_path}, error: {e}")
            return torch.zeros(num_frames, 3, 224, 224, dtype=torch.uint8)

    video_tensors = [load_video_frames(path) for path in video_paths]
    
    with torch.no_grad():
        rewards = inferencer.reward_from_frames(video_tensors, prompts, use_norm=True)
        print(rewards)
