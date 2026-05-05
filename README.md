<div align="center">

<h1>Stream-R1: <br> Reliability-Perplexity Aware Reward Distillation <br> for Streaming Video Generation</h1>

<div>
  <a href="#" target="_blank"><strong>Bin Wu</strong></a><sup>1</sup>,
  <a href="https://corleone-huang.github.io/" target="_blank"><strong>Mengqi Huang</strong></a><sup>1,&dagger;,&Dagger;</sup>,
  <a href="#" target="_blank"><strong>Shaojin Wu</strong></a><sup>3,&Dagger;</sup>,
  <a href="#" target="_blank">Weinan Jia</a><sup>1</sup>,
  <a href="#" target="_blank">Yuxin Wang</a><sup>2</sup>,
  <a href="#" target="_blank">Zhendong Mao</a><sup>1</sup>,
  <a href="#" target="_blank">Yongdong Zhang</a><sup>1</sup>
</div>

<br>

<div>
  <sup>1</sup> University of Science and Technology of China,
  <sup>2</sup> FrameX.AI,
  <sup>3</sup> Independent Researcher
</div>

<br>

<sub><sup>&dagger;</sup> Corresponding author &nbsp;&middot;&nbsp; <sup>&Dagger;</sup> Project lead</sub>

<br><br>

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://stream-r1.github.io/)
[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](#) <!-- arXiv link to be added -->
[![Models](https://img.shields.io/badge/🤗_Models-Coming_soon-yellow)](#)

</div>

## Overview

> **TL;DR**: Existing distribution-matching distillation (DMD) methods for streaming video diffusion treat every rollout, frame, and pixel as equally informative supervision. **Stream-R1** instead reweights the DMD objective along two complementary axes — *Inter-Reliability* across rollouts and *Intra-Perplexity* across spatiotemporal regions — with a single shared video reward model. The student concentrates updates where the local reward landscape has not yet flattened, converging to the teacher's high-quality mode rather than its full mixture, and surpasses the multi-step Wan2.1 teacher on VBench Total/Semantic at **23.1 FPS** with no architectural change and zero inference overhead.

## Showcase &mdash; 60-second videos

Four 60-second clips generated end-to-end by Stream-R1 at **23.1 FPS** with a single 1.3B model &mdash; no chunking, no interpolation, pure autoregressive rollout. Full set across 30 s / 60 s / 2 min / 3 min: <https://stream-r1.github.io/#duration>.

<!--
  ─────────────────────────────────────────────────────────────────────────────
  HOW TO MAKE THE FOUR PLAYERS BELOW RENDER
  ─────────────────────────────────────────────────────────────────────────────
  GitHub only renders inline videos from its own user-attachments CDN — not
  from repo-checked-in files (raw.githubusercontent.com sends Content-Disposition: attachment)
  and not from external CDNs like jsDelivr (blocked by media-src CSP).

  To populate the four players:
    1. Open this README via the pencil (Edit) icon
    2. For each "▼ drag …" line, click at its end, press Enter, then drag the
       matching templates/000X_seed0.mp4 from your file system onto the new line
    3. GitHub uploads to user-attachments and inserts a URL that auto-renders
    4. (Optional) delete the original "▼ drag …" prompt line
    5. Commit

  The committed mp4s in templates/ stay useful for anyone cloning the repo or
  for use on the project page; only the README inlines need the user-attachments URLs.
  ─────────────────────────────────────────────────────────────────────────────
-->

**Case 0000** &nbsp;&middot;&nbsp; ▼ drag `templates/0000_seed0.mp4` onto the next line, then commit

**Case 0043** &nbsp;&middot;&nbsp; ▼ drag `templates/0043_seed0.mp4` onto the next line, then commit

**Case 0069** &nbsp;&middot;&nbsp; ▼ drag `templates/0069_seed0.mp4` onto the next line, then commit

**Case 0075** &nbsp;&middot;&nbsp; ▼ drag `templates/0075_seed0.mp4` onto the next line, then commit

### Method

Stream-R1 modulates the standard DMD generator loss as

$$\mathcal{L}_{\text{Stream-R1}} \;=\; \underbrace{\exp(\beta \cdot r_{\text{final}})}_{\mathbf{W}_{\text{inter}}}\;\cdot\;\text{mean}\!\big(\underbrace{\mathbf{W}_{\text{intra}}}_{F\times H\times W}\,\odot\,\mathcal{L}_{\text{DMD}}\big)$$

with three reward-guided components, all derived from one pretrained video reward model:

1. **Inter-Reliability Weighting** — the DMD gradient `g = f_fake − f_real` varies in reliability across rollouts; we exponentially rescale each rollout's loss by `exp(β·r_final)`, so reliable rollouts dominate supervision while low-quality rollouts are attenuated.
2. **Intra-Perplexity Weighting** — back-propagates the reward model to obtain a per-pixel saliency volume `S ∈ R^{F×H×W}`, factorizes it into a temporal profile and per-frame spatial maps, and uses the product as `W_intra`. Optimization pressure concentrates on the regions and frames where the local reward landscape has not yet flattened — i.e. where further refinement yields the largest expected gain.
3. **Adaptive Reward Balancing** — tracks per-axis (VQ / MQ / TA) improvement in a sliding window and subtracts the std of per-axis deltas from the reward, keeping the three quality axes improving at similar rates.

Saliency from the three axes is fused with an adaptive softmax weighting that allocates more attention to the currently weaker axis, so a single reward signal drives both `W_inter` and `W_intra`.

### Shipped configuration (`configs/exp_stream_r1.yaml`)

| Knob | Value | Role |
|---|---|---|
| `reward_mode` | `BalancedOverall` | Inter-Reliability + Adaptive Reward Balancing |
| `spatial_reward` / `spatial_reward_pixel_grad` | `true` / `true` | Intra-Perplexity spatial (pixel-level gradient saliency) |
| `temporal_saliency_weighting` | `true` | Intra-Perplexity temporal (per-frame importance) |
| `spatial_reward_combination` | `adaptive` | adaptive saliency fusion across VQ/MQ/TA |
| `spatial_reward_min_weight` | `0.15` | spatial floor (σ\_min) |
| `temporal_saliency_min_weight` | `0.2` | temporal floor (τ\_min) |
| `full_training_steps` × `gradient_accumulation_steps` | `1000 × 8` | 8000 raw steps on 8 GPUs |

## Requirements

- NVIDIA GPU: ≥24 GB for inference, ≥80 GB for training (8 GPUs recommended).
- Linux, ≥64 GB RAM.

## Installation

```bash
git clone https://github.com/FrameX-AI/Stream-R1.git
cd Stream-R1

conda create -n stream_r1 python=3.10
conda activate stream_r1

pip install -r requirements.txt
pip install flash-attn --no-build-isolation
pip install -e .
```

## Pretrained Checkpoints

Required for **training** (teacher / reward / init):

| Model | Download |
|-------|----------|
| VideoReward | [Hugging Face](https://huggingface.co/KlingTeam/VideoReward) |
| Wan2.1-T2V-1.3B | [Hugging Face](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) |
| Wan2.1-T2V-14B | [Hugging Face](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B) |
| ODE Initialization | [Hugging Face](https://huggingface.co/gdhe17/Self-Forcing/blob/main/checkpoints/ode_init.pt) |
| **Stream-R1 (T2V-1.3B)** | **Coming soon** — pretrained weights will be released here. |

After downloading:
```
checkpoints/
├── Videoreward/
├── Wan2.1-T2V-1.3B/
├── Wan2.1-T2V-14B/
├── Stream-R1-T2V-1.3B/   # (will be added once uploaded)
└── ode_init.pt
```

Or run the helper (downloads everything except the not-yet-released Stream-R1 weights):
```bash
pip install "huggingface_hub[cli]"
bash download_checkpoints.sh
```

## Inference

> Stream-R1 weights are not yet released. Once uploaded, place them at
> `checkpoints/Stream-R1-T2V-1.3B/stream_r1.pt` (any filename works — pass it via `--checkpoint_path`). Until then, you can run inference using a checkpoint produced by your own training run (`output/<timestamp>_stream_r1/checkpoint_model_*/generator.pt`).

```bash
# 5-second video
python inference.py \
    --num_output_frames 21 \
    --config_path configs/stream_r1.yaml \
    --checkpoint_path checkpoints/Stream-R1-T2V-1.3B/stream_r1.pt \
    --output_folder videos/stream_r1-5s \
    --data_path prompts/MovieGenVideoBench_extended.txt \
    --use_ema

# 30-second video
python inference.py \
    --num_output_frames 120 \
    --config_path configs/stream_r1.yaml \
    --checkpoint_path checkpoints/Stream-R1-T2V-1.3B/stream_r1.pt \
    --output_folder videos/stream_r1-30s \
    --data_path prompts/MovieGenVideoBench_extended.txt \
    --use_ema
```

## Training

```bash
bash run_stream_r1.sh
```

The launcher reads `configs/exp_stream_r1.yaml`, runs `train.py` via `torchrun` on 8 GPUs (1000 optimizer steps × grad-accum 8 → 8000 raw steps), then renders 20 evaluation videos from the final checkpoint. Override defaults with environment variables, e.g.:

```bash
NUM_GPUS=4 CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_stream_r1.sh
```

Manual launch:
```bash
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=5235 --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_PORT train.py \
    --config_path configs/exp_stream_r1.yaml \
    --logdir logs/stream_r1 \
    --disable-wandb
```

For multi-node training, set `--nnodes`, `--node-rank`, and `--rdzv_endpoint=$MASTER_IP:$MASTER_PORT` accordingly.

## Results

### VBench (5-second, 832×480)

| Model | Params | FPS↑ | Total↑ | Quality↑ | Semantic↑ |
|-------|:------:|:----:|:------:|:--------:|:---------:|
| Wan2.1 (multi-step teacher) | 1.3B | 0.78 | 84.26 | **85.30** | 80.09 |
| LTX-Video | 1.9B | 8.98 | 80.00 | 82.30 | 70.79 |
| SkyReels-V2 | 1.3B | 0.49 | 82.67 | 84.70 | 74.53 |
| MAGI-1 | 4.5B | 0.19 | 79.18 | 82.04 | 67.74 |
| NOVA | 0.6B | 0.88 | 80.12 | 80.39 | 79.05 |
| Pyramid Flow | 2B | 6.7 | 81.72 | 84.74 | 69.62 |
| CausVid | 1.3B | 17.0 | 82.88 | 83.93 | 78.69 |
| Self Forcing | 1.3B | 17.0 | 83.80 | 84.59 | 80.64 |
| LongLive | 1.3B | 20.7 | 83.22 | 83.68 | 81.37 |
| Rolling Forcing | 1.3B | 17.5 | 81.22 | 84.08 | 69.78 |
| Reward Forcing | 1.3B | 23.1 | 84.13 | 84.84 | 81.32 |
| **Stream-R1 (Ours)** | **1.3B** | **23.1** | **84.40** | <u>85.14</u> | **81.44** |

Stream-R1 surpasses its multi-step Wan2.1 teacher on **Total** and **Semantic** while running ~30× faster, demonstrating that reward-guided distillation can push the student beyond the teacher's quality frontier. (Underlined = second best, **bold** = best.)

### VideoReward (per-axis)

| Model | Visual↑ | Dynamic↑ | Text↑ |
|-------|:-------:|:--------:|:-----:|
| SkyReels-V2 | 3.30 | 3.05 | 2.70 |
| CausVid | 4.66 | 3.16 | 3.32 |
| Self Forcing | 3.89 | 3.44 | 3.11 |
| LongLive | 4.79 | 3.81 | 3.98 |
| Reward Forcing | 4.82 | **4.18** | 4.04 |
| **Stream-R1 (Ours)** | **4.92** | <u>4.04</u> | **4.11** |

Project page and qualitative results: <https://stream-r1.github.io/>

## Citation

A BibTeX entry will be added once the arXiv preprint is online.

## Acknowledgements

Built on [CausVid](https://github.com/tianweiy/CausVid), [Self Forcing](https://github.com/guandeh17/Self-Forcing), [Wan2.1](https://github.com/Wan-Video/Wan2.1), and [VideoAlign](https://github.com/KlingTeam/VideoAlign). Stream-R1 extends the Reward Forcing codebase with the Inter-Reliability / Intra-Perplexity formulation.

## License

See [LICENSE](LICENSE).

## Contact

For questions about the project, please open a GitHub issue or reach out to the corresponding author [Mengqi Huang](https://corleone-huang.github.io/).
