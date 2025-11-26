#!/usr/bin/env python3
"""
Main training script for video world model.
"""
import argparse
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import VideoWorldModel
from src.data import create_data_loader
from src.training import Trainer, CombinedLoss


def parse_args():
    parser = argparse.ArgumentParser(description='Train video world model')
    
    # Data
    parser.add_argument('--train-dir', type=str, default='data/split/train',
                        help='Training data directory')
    parser.add_argument('--val-dir', type=str, default='data/split/val',
                        help='Validation data directory')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--T-in', type=int, default=30,
                        help='Number of input frames')
    parser.add_argument('--T-out', type=int, default=30,
                        help='Number of output frames to predict')
    parser.add_argument('--stride', type=int, default=5,
                        help='Stride for sliding window')
    parser.add_argument('--resize', type=int, nargs=2, default=[64, 64],
                        help='Resize frames to (H, W)')
    
    # Model
    parser.add_argument('--latent-dim', type=int, default=256,
                        help='Latent dimension')
    parser.add_argument('--hidden-dim', type=int, default=512,
                        help='Hidden dimension for dynamics model')
    parser.add_argument('--world-model', type=str, default='gru',
                        choices=['gru', 'lstm', 'transformer'],
                        help='World model type')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='Number of layers in dynamics model')
    parser.add_argument('--use-pretrained', action='store_true',
                        help='Use pretrained encoder (ResNet/DINOv2)')
    parser.add_argument('--pretrained-model', type=str, default='resnet50',
                        choices=['resnet50', 'resnet101', 'dinov2_vits14', 
                                'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14'],
                        help='Pretrained model name')
    parser.add_argument('--freeze-encoder', action='store_true',
                        help='Freeze encoder weights')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--gradient-clip', type=float, default=1.0,
                        help='Gradient clipping value')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'plateau'],
                        help='Learning rate scheduler')
    
    # Loss
    parser.add_argument('--recon-weight', type=float, default=1.0,
                        help='Reconstruction loss weight')
    parser.add_argument('--perceptual-weight', type=float, default=0.1,
                        help='Perceptual loss weight')
    parser.add_argument('--temporal-weight', type=float, default=0.01,
                        help='Temporal consistency loss weight')
    parser.add_argument('--recon-type', type=str, default='mse',
                        choices=['mse', 'l1'],
                        help='Reconstruction loss type')
    parser.add_argument('--no-perceptual', action='store_true',
                        help='Disable perceptual loss')
    
    # Checkpointing
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--log-dir', type=str, default='runs',
                        help='Tensorboard log directory')
    parser.add_argument('--save-every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--validate-every', type=int, default=1,
                        help='Validate every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu', 'mps'],
                        help='Device to use')
    
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
    print("Video World Model Training")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Device: {device}")
    print(f"  Train dir: {args.train_dir}")
    print(f"  Val dir: {args.val_dir}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Input frames: {args.T_in}")
    print(f"  Output frames: {args.T_out}")
    print(f"  Frame size: {args.resize}")
    print(f"  World model: {args.world_model}")
    print(f"  Latent dim: {args.latent_dim}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Use pretrained: {args.use_pretrained}")
    if args.use_pretrained:
        print(f"  Pretrained model: {args.pretrained_model}")
        print(f"  Freeze encoder: {args.freeze_encoder}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Reconstruction type: {args.recon_type}")
    print(f"  Use perceptual: {not args.no_perceptual and device == 'cuda'}")
    print()
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader = create_data_loader(
        video_dir=args.train_dir,
        T_in=args.T_in,
        T_out=args.T_out,
        batch_size=args.batch_size,
        stride=args.stride,
        resize=tuple(args.resize),
        shuffle=True,
        augment=True,
        num_workers=args.num_workers
    )
    
    val_loader = create_data_loader(
        video_dir=args.val_dir,
        T_in=args.T_in,
        T_out=args.T_out,
        batch_size=args.batch_size,
        stride=args.stride,
        resize=tuple(args.resize),
        shuffle=False,
        augment=False,
        num_workers=args.num_workers
    )
    
    print(f"Training clips: {len(train_loader.dataset)}")
    print(f"Validation clips: {len(val_loader.dataset)}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Create model
    print("\nCreating model...")
    model = VideoWorldModel(
        input_channels=3,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        output_size=args.resize[0],  # Assume square
        world_model_type=args.world_model,
        num_layers=args.num_layers,
        use_pretrained_encoder=args.use_pretrained,
        pretrained_model_name=args.pretrained_model if args.use_pretrained else None,
        freeze_encoder=args.freeze_encoder
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {num_params:,}")
    print(f"Trainable parameters: {num_trainable:,}")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Create scheduler
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=1e-6
        )
    elif args.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )
    else:
        scheduler = None
    
    # Create loss function
    use_perceptual = (not args.no_perceptual) and (device == 'cuda')
    loss_fn = CombinedLoss(
        recon_weight=args.recon_weight,
        perceptual_weight=args.perceptual_weight if use_perceptual else 0.0,
        temporal_weight=args.temporal_weight,
        recon_type=args.recon_type,
        use_perceptual=use_perceptual,
        device=device
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        num_predictions=args.T_out,
        gradient_clip=args.gradient_clip
    )
    
    # Resume from checkpoint if specified
    if args.resume is not None:
        trainer.load_checkpoint(args.resume)
    
    # Train
    print("\nStarting training...")
    print("="*70)
    trainer.train(
        num_epochs=args.epochs,
        save_every=args.save_every,
        validate_every=args.validate_every
    )
    
    print("\n🎉 Training complete!")
    print(f"Best model saved to: {Path(args.checkpoint_dir) / 'best_model.pt'}")
    print(f"Tensorboard logs: {args.log_dir}")
    print("\nTo view training progress, run:")
    print(f"  tensorboard --logdir {args.log_dir}")


if __name__ == '__main__':
    main()
