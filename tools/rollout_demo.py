#!/usr/bin/env python3
"""
Autoregressive rollout demo: visualize world model predictions.

Given the first T_in frames of a video, predict the next K frames autoregressively.
Generates side-by-side comparison: Ground Truth | Model Prediction
"""
import argparse
import torch
import cv2
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import VideoWorldModel
from src.data.dataset import VideoClipDataset


def load_model(checkpoint_path: str, device: str) -> VideoWorldModel:
    """Load trained model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model config from checkpoint if available
    # Otherwise use default values
    model = VideoWorldModel(
        input_channels=3,
        latent_dim=256,
        hidden_dim=512,
        output_size=64,
        world_model_type='gru',  # Adjust if needed
        use_pretrained_encoder=False  # Adjust if needed
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully (epoch {checkpoint['epoch']})")
    return model


@torch.no_grad()
def autoregressive_rollout(model: VideoWorldModel, 
                           context: torch.Tensor, 
                           num_frames: int,
                           device: str) -> torch.Tensor:
    """
    Autoregressively predict future frames.
    
    Args:
        model: Trained world model
        context: Initial context frames (1, T_in, C, H, W)
        num_frames: Number of frames to predict
        device: Device
        
    Returns:
        predictions: All predicted frames (1, num_frames, C, H, W)
    """
    model.eval()
    context = context.to(device)
    
    all_predictions = []
    current_context = context.clone()
    
    print(f"Rolling out {num_frames} frames autoregressively...")
    
    for i in tqdm(range(num_frames)):
        # Predict one frame
        pred, _, _ = model(current_context, num_predictions=1)  # (1, 1, C, H, W)
        all_predictions.append(pred[:, 0])  # (1, C, H, W)
        
        # Slide window: drop oldest frame, append new prediction
        current_context = torch.cat([
            current_context[:, 1:],  # Drop first frame
            pred  # Add prediction
        ], dim=1)
    
    # Stack all predictions
    predictions = torch.stack(all_predictions, dim=1)  # (1, num_frames, C, H, W)
    return predictions


def denormalize_frame(frame: np.ndarray) -> np.ndarray:
    """Convert from [-1, 1] to [0, 255] uint8."""
    frame = (frame + 1) / 2  # [-1, 1] -> [0, 1]
    frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
    return frame


def create_side_by_side_video(ground_truth: torch.Tensor,
                               predictions: torch.Tensor,
                               context_frames: torch.Tensor,
                               output_path: str,
                               fps: int = 30):
    """
    Create side-by-side comparison video.
    
    Args:
        ground_truth: Ground truth frames (1, T, C, H, W)
        predictions: Predicted frames (1, T, C, H, W)
        context_frames: Initial context frames (1, T_in, C, H, W)
        output_path: Output video path
        fps: Frames per second
    """
    # Convert to numpy: (T, H, W, C)
    gt = ground_truth[0].cpu().numpy().transpose(0, 2, 3, 1)
    pred = predictions[0].cpu().numpy().transpose(0, 2, 3, 1)
    context = context_frames[0].cpu().numpy().transpose(0, 2, 3, 1)
    
    # Denormalize
    gt = np.array([denormalize_frame(f) for f in gt])
    pred = np.array([denormalize_frame(f) for f in pred])
    context = np.array([denormalize_frame(f) for f in context])
    
    T_context = context.shape[0]
    T_pred = pred.shape[0]
    H, W, C = gt.shape[1:]
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (W * 2 + 10, H))
    
    print(f"\nCreating video with {T_context} context + {T_pred} prediction frames...")
    
    # Show context frames (both sides show ground truth with "Context" label)
    for i in tqdm(range(T_context), desc="Context frames"):
        frame = context[i]
        
        # Create side-by-side
        canvas = np.ones((H, W * 2 + 10, 3), dtype=np.uint8) * 255
        canvas[:, :W] = frame
        canvas[:, W+10:] = frame
        
        # Add labels
        cv2.putText(canvas, "Ground Truth (Context)", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(canvas, "Model Input", (W + 20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(canvas, f"Frame {i+1}/{T_context+T_pred}", (10, H - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        out.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    
    # Show predictions vs ground truth
    for i in tqdm(range(T_pred), desc="Prediction frames"):
        gt_frame = gt[i]
        pred_frame = pred[i]
        
        # Create side-by-side
        canvas = np.ones((H, W * 2 + 10, 3), dtype=np.uint8) * 255
        canvas[:, :W] = gt_frame
        canvas[:, W+10:] = pred_frame
        
        # Add labels
        cv2.putText(canvas, "Ground Truth", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(canvas, "Model Prediction", (W + 20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(canvas, f"Frame {T_context+i+1}/{T_context+T_pred}", (10, H - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Compute and display MSE
        mse = np.mean((gt_frame.astype(float) - pred_frame.astype(float)) ** 2)
        cv2.putText(canvas, f"MSE: {mse:.1f}", (W + 20, H - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        out.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    
    out.release()
    print(f"✅ Video saved to {output_path}")


def save_frame_grid(ground_truth: torch.Tensor,
                   predictions: torch.Tensor,
                   context_frames: torch.Tensor,
                   output_path: str,
                   num_display: int = 10):
    """
    Save a grid of sampled frames for quick visualization.
    
    Args:
        ground_truth: Ground truth frames (1, T, C, H, W)
        predictions: Predicted frames (1, T, C, H, W)
        context_frames: Context frames (1, T_in, C, H, W)
        output_path: Output image path
        num_display: Number of frames to display
    """
    T_pred = predictions.shape[1]
    indices = np.linspace(0, T_pred - 1, num_display, dtype=int)
    
    # Select frames
    gt_frames = ground_truth[0, indices].cpu().numpy().transpose(0, 2, 3, 1)
    pred_frames = predictions[0, indices].cpu().numpy().transpose(0, 2, 3, 1)
    
    # Denormalize
    gt_frames = np.array([denormalize_frame(f) for f in gt_frames])
    pred_frames = np.array([denormalize_frame(f) for f in pred_frames])
    
    # Create figure
    fig, axes = plt.subplots(2, num_display, figsize=(num_display * 2, 4))
    
    for i in range(num_display):
        # Ground truth
        axes[0, i].imshow(gt_frames[i])
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Ground Truth', fontsize=10)
        axes[0, i].text(5, 10, f't={indices[i]}', color='white', 
                       fontsize=8, bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
        
        # Prediction
        axes[1, i].imshow(pred_frames[i])
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Prediction', fontsize=10)
        
        # MSE
        mse = np.mean((gt_frames[i].astype(float) - pred_frames[i].astype(float)) ** 2)
        axes[1, i].text(5, 10, f'MSE={mse:.0f}', color='white', 
                       fontsize=8, bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Frame grid saved to {output_path}")
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Autoregressive rollout demo')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--video-dir', type=str, default='data/split/test',
                       help='Directory containing test videos')
    parser.add_argument('--video-idx', type=int, default=0,
                       help='Index of video to use for demo')
    parser.add_argument('--T-in', type=int, default=30,
                       help='Number of context frames')
    parser.add_argument('--num-predict', type=int, default=30,
                       help='Number of frames to predict autoregressively')
    parser.add_argument('--output-dir', type=str, default='outputs/demos',
                       help='Output directory for demo videos')
    parser.add_argument('--fps', type=int, default=30,
                       help='Output video FPS')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu', 'mps'],
                       help='Device to use')
    parser.add_argument('--resize', type=int, nargs=2, default=[64, 64],
                       help='Frame size')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    elif args.device == 'mps' and not torch.backends.mps.is_available():
        print("MPS not available, using CPU")
        device = 'cpu'
    else:
        device = args.device
    
    print("="*70)
    print("Autoregressive Rollout Demo")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Video directory: {args.video_dir}")
    print(f"Context frames: {args.T_in}")
    print(f"Predict frames: {args.num_predict}")
    print(f"Device: {device}")
    print()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = load_model(args.checkpoint, device)
    
    # Load dataset
    print(f"\nLoading video from {args.video_dir}...")
    dataset = VideoClipDataset(
        video_dir=args.video_dir,
        T_in=args.T_in,
        T_out=args.num_predict,
        stride=1000,  # Large stride to get separate videos
        resize=tuple(args.resize),
        augment=False
    )
    
    if len(dataset) == 0:
        print("❌ No videos found!")
        return
    
    print(f"Found {len(dataset)} video clips")
    
    # Get video clip
    if args.video_idx >= len(dataset):
        print(f"⚠️  Video index {args.video_idx} out of range, using 0")
        args.video_idx = 0
    
    context, target = dataset[args.video_idx]
    print(f"\nUsing video clip {args.video_idx}")
    print(f"Context shape: {context.shape}")
    print(f"Target shape: {target.shape}")
    
    # Add batch dimension
    context = context.unsqueeze(0)  # (1, T_in, C, H, W)
    target = target.unsqueeze(0)    # (1, T_out, C, H, W)
    
    # Run autoregressive rollout
    predictions = autoregressive_rollout(
        model, context, args.num_predict, device
    )
    
    # Create visualizations
    video_path = output_dir / f'rollout_video_{args.video_idx}.mp4'
    grid_path = output_dir / f'rollout_grid_{args.video_idx}.png'
    
    create_side_by_side_video(
        target, predictions, context, 
        str(video_path), fps=args.fps
    )
    
    save_frame_grid(
        target, predictions, context,
        str(grid_path), num_display=10
    )
    
    # Compute metrics
    mse = torch.mean((target.cpu() - predictions.cpu()) ** 2).item()
    mae = torch.mean(torch.abs(target.cpu() - predictions.cpu())).item()
    
    print("\n" + "="*70)
    print("Rollout Metrics:")
    print(f"  MSE: {mse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print("="*70)
    print(f"\n✅ Demo complete!")
    print(f"Video: {video_path}")
    print(f"Grid: {grid_path}")


if __name__ == '__main__':
    main()
