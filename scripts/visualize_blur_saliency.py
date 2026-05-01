#!/usr/bin/env python3
"""
Visualize real gradient-based spatiotemporal weights on degraded video.

Applies localized Gaussian blur to specified frames/regions to simulate
teacher quality deficiencies, then runs the reward model's gradient
backpropagation to obtain *actual* saliency maps. Produces a paper figure:
  Row 1: Video frames (with blur regions visible)
  Row 2: Spatial weight heatmaps overlaid on frames
  Row 3: Temporal weight bar chart (per-frame)

Usage:
    # With real reward model:
    python scripts/visualize_blur_saliency.py \
        --video_path path/to/video.mp4 \
        --prompt "A cheetah running across the savanna" \
        --reward_ckpt checkpoints/Videoreward \
        --output_dir viz_blur_saliency

    # Dummy mode (layout testing, no model needed):
    python scripts/visualize_blur_saliency.py \
        --video_path path/to/video.mp4 \
        --dummy --output_dir viz_blur_saliency

    # Custom blur config from JSON:
    python scripts/visualize_blur_saliency.py \
        --video_path path/to/video.mp4 \
        --blur_config blur_specs.json \
        --reward_ckpt checkpoints/Videoreward
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as nnf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter as scipy_gaussian

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts.visualize_spatial_saliency import (
    load_video_frames,
    compute_saliency_with_model,
    compute_saliency_dummy,
    adaptive_combine,
    postprocess_factored_weight,
)

METRICS = ("MQ", "VQ", "TA")

# Weight colormap: light = low weight, deep navy = high weight
WEIGHT_CMAP = LinearSegmentedColormap.from_list("wt", [
    (0.00, "#f0f4fa"),
    (0.30, "#a0bde0"),
    (0.60, "#4a82c4"),
    (1.00, "#0a2e66"),
])


# ---------------------------------------------------------------------------
# Default blur specifications
# ---------------------------------------------------------------------------
def default_blur_specs(T, clean_frames=20, blur_sigma=20,
                       sy_min=0.06, sy_max=0.28,
                       sx_min=0.08, sx_max=0.35):
    """Generate default blur specs for a video of T frames.

    Returns a dict: {frame_index: [(cy, cx, sy, sx, blur_sigma), ...]}
    Coordinates are in [0, 1] normalized space.
    First `clean_frames` frames are clean (no blur).
    Remaining frames get center-bottom blur with FIXED intensity but
    a region size that linearly grows from (sy_min, sx_min) to
    (sy_max, sx_max).
    """
    specs = {}
    blur_start = min(clean_frames, T)
    blur_count = T - blur_start
    for i in range(blur_start, T):
        ratio = (i - blur_start) / max(blur_count - 1, 1)
        sy = sy_min + ratio * (sy_max - sy_min)
        sx = sx_min + ratio * (sx_max - sx_min)
        specs[i] = [(0.65, 0.50, sy, sx, blur_sigma)]
    return specs


# ---------------------------------------------------------------------------
# Blur application
# ---------------------------------------------------------------------------
def apply_blur_to_video(video_tensor, blur_specs):
    """Apply localized Gaussian blur to specified frames and regions.

    Args:
        video_tensor: [T, C, H, W] in [0, 1]
        blur_specs: dict {frame_idx: [(cy, cx, sy, sx, blur_sigma), ...]}

    Returns:
        blurred_video: [T, C, H, W]
        blur_masks: dict {frame_idx: [H, W] mask in [0, 1]}
    """
    T, C, H, W = video_tensor.shape
    blurred = video_tensor.clone()
    blur_masks = {}

    Y = torch.linspace(0, 1, H).unsqueeze(1).expand(H, W)
    X = torch.linspace(0, 1, W).unsqueeze(0).expand(H, W)

    for frame_idx, regions in blur_specs.items():
        frame_idx = int(frame_idx)
        if frame_idx >= T:
            continue

        frame_np = video_tensor[frame_idx].permute(1, 2, 0).numpy()  # [H, W, C]
        total_mask = torch.zeros(H, W)

        for cy, cx, sy, sx, sigma in regions:
            mask = torch.exp(-0.5 * (((Y - cy) / sy) ** 2 + ((X - cx) / sx) ** 2))
            total_mask = torch.maximum(total_mask, mask)

            blurred_np = scipy_gaussian(frame_np, sigma=(sigma, sigma, 0))
            mask_np = mask.numpy()[:, :, None]
            frame_np = frame_np * (1 - mask_np) + blurred_np * mask_np

        blurred[frame_idx] = torch.from_numpy(frame_np).permute(2, 0, 1).float().clamp(0, 1)
        blur_masks[frame_idx] = total_mask.clamp(0, 1)

    return blurred, blur_masks


# ---------------------------------------------------------------------------
# Visualization: paper figure
# ---------------------------------------------------------------------------
def render_paper_figure(video_tensor, blurred_tensor, spatial_weight,
                        temporal_weight, blur_masks, blur_specs,
                        display_indices, save_path, reward_dict=None,
                        cols_per_page=8, spread_sigma=0.0, gamma=1.0):
    """Render paginated paper figures covering ALL frames.

    Each page has `cols_per_page` frames with:
      Row 1: Video frames (blurred version, red outline on degraded)
      Row 2: Spatial weight heatmaps
    A separate temporal weight bar chart covering all frames is saved once.
    """
    T, C, H, W = blurred_tensor.shape
    F_lat = spatial_weight.shape[0]
    tw = temporal_weight.numpy()

    # Interpolate temporal weights to video frame count
    tw_interp = nnf.interpolate(
        torch.from_numpy(tw).float().unsqueeze(0).unsqueeze(0),
        size=T, mode="linear", align_corners=True
    ).squeeze().numpy()

    save_dir = os.path.splitext(save_path)[0] if save_path else "paper_figure"
    os.makedirs(save_dir, exist_ok=True)

    all_frames = display_indices
    frames_per_page = cols_per_page * 2  # 2 rows x cols_per_page
    n_pages = int(np.ceil(len(all_frames) / frames_per_page))

    for page in range(n_pages):
        start = page * frames_per_page
        end = min(start + frames_per_page, len(all_frames))
        page_frames = all_frames[start:end]

        # Split into top row and bottom row
        top_frames = page_frames[:cols_per_page]
        bot_frames = page_frames[cols_per_page:]
        n_rows = 4 if bot_frames else 2  # 2 rows per group (frame + weight)

        fig, axes = plt.subplots(n_rows, cols_per_page,
                                  figsize=(3.2 * cols_per_page, 3.2 * n_rows),
                                  squeeze=False, facecolor="white")

        def _fill_pair(row_frame, row_weight, frames_list):
            for col in range(cols_per_page):
                if col < len(frames_list):
                    fi = frames_list[col]
                    frame = blurred_tensor[fi].permute(1, 2, 0).cpu().numpy()
                    axes[row_frame, col].imshow(np.clip(frame, 0, 1))

                    is_blurred = fi in blur_specs
                    if is_blurred:
                        for sp in axes[row_frame, col].spines.values():
                            sp.set_edgecolor("#e74c3c")
                            sp.set_linewidth(3)
                            sp.set_visible(True)
                    else:
                        for sp in axes[row_frame, col].spines.values():
                            sp.set_edgecolor("#bbb")
                            sp.set_linewidth(1.5)

                    label = f"Frame {fi}" + (" [degraded]" if is_blurred else "")
                    axes[row_frame, col].set_title(
                        label, fontsize=9, fontweight="bold",
                        color="#c0392b" if is_blurred else "#2d3544", pad=4)

                    si = int(round(fi / max(T - 1, 1) * (F_lat - 1)))
                    sw = spatial_weight[si]
                    sw_up = nnf.interpolate(
                        sw.unsqueeze(0).unsqueeze(0).float(),
                        size=(H, W), mode="bilinear", align_corners=False
                    ).squeeze().numpy()
                    if spread_sigma > 0:
                        sw_up = scipy_gaussian(sw_up, sigma=spread_sigma)
                    sw_min, sw_max = sw_up.min(), sw_up.max()
                    if sw_max - sw_min > 1e-6:
                        sw_disp = (sw_up - sw_min) / (sw_max - sw_min)
                    else:
                        sw_disp = np.ones_like(sw_up) * 0.5
                    if gamma != 1.0:
                        sw_disp = np.power(sw_disp, gamma)
                    axes[row_weight, col].imshow(
                        sw_disp, cmap=WEIGHT_CMAP, vmin=0, vmax=1,
                        aspect="auto", interpolation="bilinear")
                    for sp in axes[row_weight, col].spines.values():
                        sp.set_edgecolor("#bbb")
                        sp.set_linewidth(1.5)
                else:
                    axes[row_frame, col].axis("off")
                    axes[row_weight, col].axis("off")

                axes[row_frame, col].set_xticks([])
                axes[row_frame, col].set_yticks([])
                axes[row_weight, col].set_xticks([])
                axes[row_weight, col].set_yticks([])

        _fill_pair(0, 1, top_frames)
        if bot_frames:
            _fill_pair(2, 3, bot_frames)

        axes[0, 0].set_ylabel("Frame", fontsize=10, fontweight="bold")
        axes[1, 0].set_ylabel("Spatial\nWeight", fontsize=10, fontweight="bold")
        if n_rows == 4:
            axes[2, 0].set_ylabel("Frame", fontsize=10, fontweight="bold")
            axes[3, 0].set_ylabel("Spatial\nWeight", fontsize=10, fontweight="bold")

        plt.tight_layout(h_pad=0.4, w_pad=0.3)
        page_path = os.path.join(save_dir, f"frames_{all_frames[start]:03d}-{all_frames[end-1]:03d}.png")
        plt.savefig(page_path, dpi=200, bbox_inches="tight",
                    pad_inches=0.06, facecolor="white")
        plt.close(fig)

    # --- Temporal weight bar chart (one image for all frames) ---
    fig, ax = plt.subplots(figsize=(16, 3.5), facecolor="white")
    bar_colors = []
    for i in range(T):
        if i in blur_specs:
            bar_colors.append("#e74c3c")
        else:
            bar_colors.append(WEIGHT_CMAP(tw_interp[i] / tw_interp.max()))

    ax.bar(range(T), tw_interp, color=bar_colors,
           edgecolor="none", width=1.0, alpha=0.85)
    ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.set_xlim(-0.5, T - 0.5)
    ax.set_ylim(0, tw_interp.max() * 1.2)
    ax.set_xlabel("Frame Index", fontsize=11)
    ax.set_ylabel("Temporal Weight", fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if reward_dict:
        reward_str = "  ".join(f"{m}={reward_dict[m]:.2f}" for m in METRICS)
        ax.set_title(f"Temporal Weight Profile    (Reward: {reward_str})",
                     fontsize=11, pad=8)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "temporal_weights.png"),
                dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved {n_pages} pages + temporal chart to {save_dir}/")


def render_overlay_figure(video_tensor, blurred_tensor, spatial_weight,
                          combined_sal, display_indices, save_path,
                          cols_per_page=4, spread_sigma=0.0, gamma=1.0):
    """Render paginated frames with saliency heatmap overlaid (jet colormap).
    Each page: 2 groups of (frame row + saliency row), cols_per_page columns."""
    T, C, H, W = blurred_tensor.shape
    F_lat = spatial_weight.shape[0]

    save_dir = os.path.splitext(save_path)[0] if save_path else "overlay_figure"
    os.makedirs(save_dir, exist_ok=True)

    all_frames = display_indices
    frames_per_page = cols_per_page * 2
    n_pages = int(np.ceil(len(all_frames) / frames_per_page))

    for page in range(n_pages):
        start = page * frames_per_page
        end = min(start + frames_per_page, len(all_frames))
        page_frames = all_frames[start:end]

        top_frames = page_frames[:cols_per_page]
        bot_frames = page_frames[cols_per_page:]
        n_rows = 4 if bot_frames else 2

        fig, axes = plt.subplots(n_rows, cols_per_page,
                                  figsize=(3.5 * cols_per_page, 3.2 * n_rows),
                                  squeeze=False, facecolor="white")

        def _fill_pair(row_frame, row_sal, frames_list):
            for col in range(cols_per_page):
                if col < len(frames_list):
                    fi = frames_list[col]
                    frame = blurred_tensor[fi].permute(1, 2, 0).cpu().numpy()
                    si = int(round(fi / max(T - 1, 1) * (F_lat - 1)))

                    sal = combined_sal[si]
                    sal_up = nnf.interpolate(
                        sal.unsqueeze(0).unsqueeze(0).float(),
                        size=(H, W), mode="bilinear", align_corners=False
                    ).squeeze().numpy()
                    if spread_sigma > 0:
                        sal_up = scipy_gaussian(sal_up, sigma=spread_sigma)
                    sal_norm = sal_up - sal_up.min()
                    if sal_norm.max() > 1e-8:
                        sal_norm = sal_norm / sal_norm.max()
                    if gamma != 1.0:
                        sal_norm = np.power(sal_norm, gamma)

                    heatmap = plt.get_cmap("jet")(sal_norm)[:, :, :3]
                    overlay = 0.45 * frame + 0.55 * heatmap

                    axes[row_frame, col].imshow(np.clip(frame, 0, 1))
                    axes[row_frame, col].set_title(f"Frame {fi}", fontsize=9,
                                                    fontweight="bold")
                    axes[row_sal, col].imshow(np.clip(overlay, 0, 1))
                else:
                    axes[row_frame, col].axis("off")
                    axes[row_sal, col].axis("off")

                axes[row_frame, col].set_xticks([])
                axes[row_frame, col].set_yticks([])
                axes[row_sal, col].set_xticks([])
                axes[row_sal, col].set_yticks([])

        _fill_pair(0, 1, top_frames)
        if bot_frames:
            _fill_pair(2, 3, bot_frames)

        axes[0, 0].set_ylabel("Frame", fontsize=10, fontweight="bold")
        axes[1, 0].set_ylabel("Saliency", fontsize=10, fontweight="bold")
        if n_rows == 4:
            axes[2, 0].set_ylabel("Frame", fontsize=10, fontweight="bold")
            axes[3, 0].set_ylabel("Saliency", fontsize=10, fontweight="bold")

        plt.tight_layout(h_pad=0.3, w_pad=0.2)
        page_path = os.path.join(save_dir, f"overlay_{all_frames[start]:03d}-{all_frames[end-1]:03d}.png")
        plt.savefig(page_path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    print(f"Saved {n_pages} overlay pages to {save_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Visualize real gradient saliency on artificially degraded video")
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--reward_ckpt", type=str, default="checkpoints/Videoreward")
    parser.add_argument("--output_dir", type=str, default="viz_blur_saliency")
    parser.add_argument("--method", type=str, default="pixel_grad",
                        choices=["pixel_grad", "gradient", "attention"])
    parser.add_argument("--dummy", action="store_true",
                        help="Use synthetic gradients (no model needed)")
    parser.add_argument("--blur_config", type=str, default=None,
                        help="JSON file with blur specs (optional)")
    parser.add_argument("--max_frames", type=int, default=81)
    parser.add_argument("--latent_shape", type=int, nargs=3, default=[21, 60, 104],
                        metavar=("F", "H", "W"))
    parser.add_argument("--display_frames", type=int, nargs="+", default=None,
                        help="Frame indices to display (default: all frames)")
    parser.add_argument("--cols_per_page", type=int, default=4,
                        help="Number of frames per row per page (default: 4)")
    # Weight hyperparameters
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--min_weight", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--temporal_strength", type=float, default=1.0)
    parser.add_argument("--temporal_min_weight", type=float, default=0.20)
    # Display enhancement
    parser.add_argument("--enhance", type=str, nargs="*", default=["spread", "gamma", "floor"],
                        choices=["spread", "gamma", "floor"],
                        help="Which enhancements to enable (default: all three). "
                             "spread=Gaussian blur on weight map; "
                             "gamma=power-law compression; "
                             "floor=lower min_weight to 0.05")
    parser.add_argument("--spread_sigma", type=float, default=15.0,
                        help="Gaussian spread sigma (only if 'spread' in --enhance)")
    parser.add_argument("--gamma", type=float, default=0.4,
                        help="Gamma exponent, <1 enlarges highlights (only if 'gamma' in --enhance)")
    args = parser.parse_args()

    # Apply enhancement toggles
    enhance_set = set(args.enhance) if args.enhance else set()
    if "spread" not in enhance_set:
        args.spread_sigma = 0.0
    if "gamma" not in enhance_set:
        args.gamma = 1.0
    if "floor" not in enhance_set:
        args.min_weight = 0.15  # restore original default
    active = [e for e in ["spread", "gamma", "floor"] if e in enhance_set]
    print(f"  Enhancements: {', '.join(active) if active else 'none'}"
          f"  (spread_sigma={args.spread_sigma}, gamma={args.gamma}, min_weight={args.min_weight})")

    os.makedirs(args.output_dir, exist_ok=True)
    latent_shape = tuple(args.latent_shape)

    # --- Auto prompt ---
    if args.prompt is None:
        args.prompt = os.path.splitext(os.path.basename(args.video_path))[0].replace("_", " ")
        print(f"  Auto prompt: \"{args.prompt}\"")

    # --- Load video ---
    print(f"Loading video: {args.video_path}")
    video_tensor = load_video_frames(args.video_path, max_frames=args.max_frames)
    T, C, H, W = video_tensor.shape
    print(f"  Video: {T} frames, {H}x{W}")

    # --- Load or generate blur specs ---
    if args.blur_config:
        with open(args.blur_config) as f:
            blur_specs_raw = json.load(f)
        blur_specs = {int(k): v for k, v in blur_specs_raw.items()}
        print(f"  Loaded blur config from {args.blur_config}")
    else:
        blur_specs = default_blur_specs(T)
        print(f"  Using default blur specs for {T} frames")

    print(f"  Blur applied to frames: {sorted(blur_specs.keys())}")

    # --- Apply blur ---
    blurred_tensor, blur_masks = apply_blur_to_video(video_tensor, blur_specs)

    # --- Compute saliency on blurred video ---
    if args.dummy:
        print("  Using DUMMY saliency (synthetic gradients)")
        saliency_result = compute_saliency_dummy(blurred_tensor, latent_shape)
    else:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from videoalign.wan_inference import VideoVLMRewardInference

        ckpt_path = args.reward_ckpt
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(project_root, ckpt_path)

        print(f"  Loading reward model: {ckpt_path}")
        inferencer = VideoVLMRewardInference(ckpt_path, device="cuda")
        inferencer.model.requires_grad_(False)

        saliency_result = compute_saliency_with_model(
            inferencer, blurred_tensor.to("cuda"), args.prompt,
            args.method, latent_shape)

    reward_dict = saliency_result["reward_dict"]
    interpolated = saliency_result["interpolated"]
    print(f"  Rewards: " + "  ".join(f"{m}={reward_dict[m]:.3f}" for m in METRICS))

    # --- Adaptive combine + factored decomposition ---
    combined, adap_weights = adaptive_combine(
        interpolated, reward_dict, METRICS, temperature=args.temperature)
    print(f"  Adaptive weights: " +
          "  ".join(f"{m}={adap_weights[i].item():.3f}" for i, m in enumerate(METRICS)))

    factored, temporal_weight, spatial_weight = postprocess_factored_weight(
        combined, args.strength, args.min_weight,
        args.temporal_strength, args.temporal_min_weight)

    tw = temporal_weight.numpy()
    print(f"  Temporal weight: min={tw.min():.3f}  max={tw.max():.3f}  std={tw.std():.3f}")

    # --- Select display frames: default = ALL frames ---
    if args.display_frames is not None:
        display_indices = args.display_frames
    else:
        display_indices = list(range(T))

    print(f"  Display frames: {len(display_indices)} frames (0-{T-1})")

    # --- Render figures (paginated, cols_per_page frames per image) ---
    render_paper_figure(
        video_tensor, blurred_tensor, spatial_weight,
        temporal_weight, blur_masks, blur_specs,
        display_indices,
        save_path=os.path.join(args.output_dir, "paper_figure.png"),
        reward_dict=reward_dict,
        cols_per_page=args.cols_per_page,
        spread_sigma=args.spread_sigma,
        gamma=args.gamma,
    )

    render_overlay_figure(
        video_tensor, blurred_tensor, spatial_weight, combined,
        display_indices,
        save_path=os.path.join(args.output_dir, "overlay_figure.png"),
        cols_per_page=args.cols_per_page,
        spread_sigma=args.spread_sigma,
        gamma=args.gamma,
    )

    # --- Save blur specs for reproducibility ---
    specs_serializable = {str(k): v for k, v in blur_specs.items()}
    with open(os.path.join(args.output_dir, "blur_specs.json"), "w") as f:
        json.dump(specs_serializable, f, indent=2)

    print(f"\nAll outputs saved to: {args.output_dir}/")
    print(f"  paper_figure/      - paginated frames + spatial weights ({args.cols_per_page}/page)")
    print(f"  paper_figure/temporal_weights.png - temporal weight bar chart")
    print(f"  overlay_figure/    - paginated saliency overlays")
    print(f"  blur_specs.json    - blur configuration (reproducibility)")


if __name__ == "__main__":
    main()
