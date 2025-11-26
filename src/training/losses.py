"""
Loss functions for video world model training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import torchvision.models as models


class ReconstructionLoss(nn.Module):
    """
    Basic reconstruction loss: MSE or L1 between predicted and target frames.
    """
    
    def __init__(self, loss_type: str = 'mse'):
        """
        Args:
            loss_type: 'mse' or 'l1'
        """
        super().__init__()
        self.loss_type = loss_type
        
        if loss_type == 'mse':
            self.loss_fn = nn.MSELoss()
        elif loss_type == 'l1':
            self.loss_fn = nn.L1Loss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted frames (B, T, C, H, W)
            target: Target frames (B, T, C, H, W)
            
        Returns:
            loss: Scalar loss value
        """
        return self.loss_fn(pred, target)


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using pretrained VGG features.
    Measures high-level feature similarity rather than pixel-wise similarity.
    """
    
    def __init__(self, feature_layers: list = None, device: str = 'cpu'):
        """
        Args:
            feature_layers: Which VGG layers to use for feature extraction
            device: Device to run VGG on
        """
        super().__init__()
        
        if feature_layers is None:
            feature_layers = [3, 8, 15, 22]  # relu1_2, relu2_2, relu3_3, relu4_3
        
        self.feature_layers = feature_layers
        
        # Load pretrained VGG16
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.vgg = vgg.eval().to(device)
        
        # Freeze VGG parameters
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        # Normalization for ImageNet
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize from [-1, 1] to ImageNet normalization."""
        # Convert from [-1, 1] to [0, 1]
        x = (x + 1) / 2
        # Apply ImageNet normalization
        return (x - self.mean) / self.std
    
    def _extract_features(self, x: torch.Tensor) -> list:
        """Extract features from specified VGG layers."""
        features = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.feature_layers:
                features.append(x)
        return features
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted frames (B, T, C, H, W)
            target: Target frames (B, T, C, H, W)
            
        Returns:
            loss: Perceptual loss value
        """
        B, T, C, H, W = pred.shape
        
        # Reshape to (B*T, C, H, W)
        pred = pred.view(B * T, C, H, W)
        target = target.view(B * T, C, H, W)
        
        # Normalize
        pred = self._normalize(pred)
        target = self._normalize(target)
        
        # Extract features
        pred_features = self._extract_features(pred)
        target_features = self._extract_features(target)
        
        # Compute feature-wise loss
        loss = 0
        for pred_feat, target_feat in zip(pred_features, target_features):
            loss += F.mse_loss(pred_feat, target_feat)
        
        return loss / len(pred_features)


class TemporalConsistencyLoss(nn.Module):
    """
    Temporal consistency loss: penalizes abrupt changes between consecutive frames.
    Encourages smooth motion.
    """
    
    def __init__(self, loss_type: str = 'l1'):
        """
        Args:
            loss_type: 'l1' or 'l2' for computing frame differences
        """
        super().__init__()
        self.loss_type = loss_type
    
    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted frames (B, T, C, H, W)
            
        Returns:
            loss: Temporal consistency loss
        """
        # Compute difference between consecutive frames
        diff = pred[:, 1:] - pred[:, :-1]  # (B, T-1, C, H, W)
        
        if self.loss_type == 'l1':
            return torch.mean(torch.abs(diff))
        elif self.loss_type == 'l2':
            return torch.mean(diff ** 2)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


class CombinedLoss(nn.Module):
    """
    Combined loss function with configurable weights.
    """
    
    def __init__(self,
                 recon_weight: float = 1.0,
                 perceptual_weight: float = 0.1,
                 temporal_weight: float = 0.01,
                 recon_type: str = 'mse',
                 use_perceptual: bool = True,
                 device: str = 'cpu'):
        """
        Args:
            recon_weight: Weight for reconstruction loss
            perceptual_weight: Weight for perceptual loss
            temporal_weight: Weight for temporal consistency loss
            recon_type: 'mse' or 'l1' for reconstruction
            use_perceptual: Whether to use perceptual loss (requires GPU)
            device: Device for perceptual loss
        """
        super().__init__()
        
        self.recon_weight = recon_weight
        self.perceptual_weight = perceptual_weight
        self.temporal_weight = temporal_weight
        self.use_perceptual = use_perceptual
        
        # Initialize loss functions
        self.recon_loss = ReconstructionLoss(recon_type)
        self.temporal_loss = TemporalConsistencyLoss('l1')
        
        if use_perceptual:
            self.perceptual_loss = PerceptualLoss(device=device)
        else:
            self.perceptual_loss = None
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple:
        """
        Args:
            pred: Predicted frames (B, T, C, H, W)
            target: Target frames (B, T, C, H, W)
            
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual losses
        """
        # Reconstruction loss
        recon = self.recon_loss(pred, target)
        
        # Temporal consistency loss
        temporal = self.temporal_loss(pred)
        
        # Perceptual loss
        if self.use_perceptual and self.perceptual_loss is not None:
            perceptual = self.perceptual_loss(pred, target)
        else:
            perceptual = torch.tensor(0.0, device=pred.device)
        
        # Combined loss
        total = (self.recon_weight * recon +
                self.perceptual_weight * perceptual +
                self.temporal_weight * temporal)
        
        loss_dict = {
            'total': total.item(),
            'reconstruction': recon.item(),
            'perceptual': perceptual.item() if isinstance(perceptual, torch.Tensor) else 0.0,
            'temporal': temporal.item()
        }
        
        return total, loss_dict


if __name__ == '__main__':
    """Test loss functions."""
    print("Testing loss functions...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Create dummy data
    B, T, C, H, W = 2, 10, 3, 64, 64
    pred = torch.randn(B, T, C, H, W, device=device)
    target = torch.randn(B, T, C, H, W, device=device)
    
    print(f"\nInput shape: {pred.shape}")
    
    # Test reconstruction loss
    print("\n" + "="*50)
    print("Testing ReconstructionLoss...")
    recon_loss_mse = ReconstructionLoss('mse')
    recon_loss_l1 = ReconstructionLoss('l1')
    
    loss_mse = recon_loss_mse(pred, target)
    loss_l1 = recon_loss_l1(pred, target)
    
    print(f"MSE Loss: {loss_mse.item():.4f}")
    print(f"L1 Loss: {loss_l1.item():.4f}")
    
    # Test temporal consistency loss
    print("\n" + "="*50)
    print("Testing TemporalConsistencyLoss...")
    temporal_loss = TemporalConsistencyLoss('l1')
    loss = temporal_loss(pred)
    print(f"Temporal Loss: {loss.item():.4f}")
    
    # Test perceptual loss (only if GPU available)
    if device == 'cuda':
        print("\n" + "="*50)
        print("Testing PerceptualLoss...")
        perceptual_loss = PerceptualLoss(device=device)
        loss = perceptual_loss(pred, target)
        print(f"Perceptual Loss: {loss.item():.4f}")
    else:
        print("\n(Skipping PerceptualLoss test - GPU required)")
    
    # Test combined loss
    print("\n" + "="*50)
    print("Testing CombinedLoss...")
    combined_loss = CombinedLoss(
        recon_weight=1.0,
        perceptual_weight=0.1,
        temporal_weight=0.01,
        use_perceptual=(device == 'cuda'),
        device=device
    )
    
    total, loss_dict = combined_loss(pred, target)
    print(f"Total Loss: {loss_dict['total']:.4f}")
    print(f"  - Reconstruction: {loss_dict['reconstruction']:.4f}")
    print(f"  - Perceptual: {loss_dict['perceptual']:.4f}")
    print(f"  - Temporal: {loss_dict['temporal']:.4f}")
    
    print("\n✅ All loss tests passed!")
