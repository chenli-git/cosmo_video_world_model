"""
Trainer class for video world model.
"""
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Optional, Dict
import time
from tqdm import tqdm

from .losses import CombinedLoss


class Trainer:
    """
    Trainer for video world model.
    Handles training loop, validation, checkpointing, and logging.
    """
    
    def __init__(self,
                 model: nn.Module,
                 train_loader,
                 val_loader,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 loss_fn: Optional[nn.Module] = None,
                 device: str = 'cuda',
                 checkpoint_dir: str = 'checkpoints',
                 log_dir: str = 'runs',
                 num_predictions: int = 30,
                 gradient_clip: float = 1.0):
        """
        Args:
            model: Video world model
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer (default: AdamW)
            scheduler: Learning rate scheduler (default: CosineAnnealingLR)
            loss_fn: Loss function (default: CombinedLoss)
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory for tensorboard logs
            num_predictions: Number of frames to predict
            gradient_clip: Gradient clipping value
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_predictions = num_predictions
        self.gradient_clip = gradient_clip
        
        # Optimizer
        if optimizer is None:
            self.optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        else:
            self.optimizer = optimizer
        
        # Scheduler
        if scheduler is None:
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=100, eta_min=1e-6)
        else:
            self.scheduler = scheduler
        
        # Loss function
        if loss_fn is None:
            use_perceptual = (device == 'cuda')  # Only use perceptual on GPU
            self.loss_fn = CombinedLoss(
                recon_weight=1.0,
                perceptual_weight=0.1 if use_perceptual else 0.0,
                temporal_weight=0.01,
                use_perceptual=use_perceptual,
                device=device
            )
        else:
            self.loss_fn = loss_fn
        
        # Checkpointing
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Logging
        self.writer = SummaryWriter(log_dir)
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        total_recon_loss = 0
        total_perceptual_loss = 0
        total_temporal_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch}')
        for context, target in pbar:
            # Move to device
            context = context.to(self.device)  # (B, T_in, C, H, W)
            target = target.to(self.device)    # (B, T_out, C, H, W)
            
            # Forward pass
            pred, _, _ = self.model(context, self.num_predictions)  # (B, T_out, C, H, W)
            
            # Compute loss
            loss, loss_dict = self.loss_fn(pred, target)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss_dict['total']
            total_recon_loss += loss_dict['reconstruction']
            total_perceptual_loss += loss_dict['perceptual']
            total_temporal_loss += loss_dict['temporal']
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'recon': f"{loss_dict['reconstruction']:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            # Log to tensorboard
            if self.global_step % 10 == 0:
                self.writer.add_scalar('train/loss', loss_dict['total'], self.global_step)
                self.writer.add_scalar('train/recon_loss', loss_dict['reconstruction'], self.global_step)
                self.writer.add_scalar('train/perceptual_loss', loss_dict['perceptual'], self.global_step)
                self.writer.add_scalar('train/temporal_loss', loss_dict['temporal'], self.global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
        
        # Compute epoch metrics
        metrics = {
            'loss': total_loss / num_batches,
            'recon_loss': total_recon_loss / num_batches,
            'perceptual_loss': total_perceptual_loss / num_batches,
            'temporal_loss': total_temporal_loss / num_batches,
        }
        
        return metrics
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate on validation set."""
        self.model.eval()
        
        total_loss = 0
        total_recon_loss = 0
        total_perceptual_loss = 0
        total_temporal_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.val_loader, desc='Validation')
        for context, target in pbar:
            # Move to device
            context = context.to(self.device)
            target = target.to(self.device)
            
            # Forward pass
            pred, _, _ = self.model(context, self.num_predictions)
            
            # Compute loss
            loss, loss_dict = self.loss_fn(pred, target)
            
            # Update metrics
            total_loss += loss_dict['total']
            total_recon_loss += loss_dict['reconstruction']
            total_perceptual_loss += loss_dict['perceptual']
            total_temporal_loss += loss_dict['temporal']
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'recon': f"{loss_dict['reconstruction']:.4f}"
            })
        
        # Compute metrics
        metrics = {
            'loss': total_loss / num_batches,
            'recon_loss': total_recon_loss / num_batches,
            'perceptual_loss': total_perceptual_loss / num_batches,
            'temporal_loss': total_temporal_loss / num_batches,
        }
        
        # Log to tensorboard
        self.writer.add_scalar('val/loss', metrics['loss'], self.epoch)
        self.writer.add_scalar('val/recon_loss', metrics['recon_loss'], self.epoch)
        self.writer.add_scalar('val/perceptual_loss', metrics['perceptual_loss'], self.epoch)
        self.writer.add_scalar('val/temporal_loss', metrics['temporal_loss'], self.epoch)
        
        return metrics
    
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Save checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
        }
        
        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Resuming from epoch {self.epoch}, step {self.global_step}")
    
    def train(self, num_epochs: int, save_every: int = 10, validate_every: int = 1):
        """
        Main training loop.
        
        Args:
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
            validate_every: Validate every N epochs
        """
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Training batches: {len(self.train_loader)}")
        print(f"Validation batches: {len(self.val_loader)}")
        print("="*70)
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Print training metrics
            print(f"\nEpoch {epoch} Training:")
            print(f"  Loss: {train_metrics['loss']:.4f}")
            print(f"  Recon: {train_metrics['recon_loss']:.4f}")
            print(f"  Perceptual: {train_metrics['perceptual_loss']:.4f}")
            print(f"  Temporal: {train_metrics['temporal_loss']:.4f}")
            
            # Validate
            if epoch % validate_every == 0:
                val_metrics = self.validate()
                
                print(f"Epoch {epoch} Validation:")
                print(f"  Loss: {val_metrics['loss']:.4f}")
                print(f"  Recon: {val_metrics['recon_loss']:.4f}")
                print(f"  Perceptual: {val_metrics['perceptual_loss']:.4f}")
                print(f"  Temporal: {val_metrics['temporal_loss']:.4f}")
                
                # Check if best model
                is_best = val_metrics['loss'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_metrics['loss']
                    print(f"  🎉 New best validation loss: {self.best_val_loss:.4f}")
                
                # Save checkpoint
                if epoch % save_every == 0 or is_best:
                    self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt', is_best=is_best)
            
            # Step scheduler
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_metrics['loss'])
            else:
                self.scheduler.step()
            
            print("="*70)
        
        # Training complete
        elapsed_time = time.time() - start_time
        print(f"\n✅ Training complete!")
        print(f"Total time: {elapsed_time/3600:.2f} hours")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # Close tensorboard writer
        self.writer.close()
