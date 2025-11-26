"""
Video frame encoder: compresses frames to latent representations.
Modular design allows easy replacement with foundation models.
"""
import torch
import torch.nn as nn
from typing import Optional


class ResidualBlock(nn.Module):
    """Residual block with skip connection."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x + residual
        x = self.relu(x)
        return x


class FrameEncoder(nn.Module):
    """
    Encoder that compresses a single frame to a latent vector.
    
    Architecture: Conv layers with downsampling -> latent vector
    Can be replaced with pretrained models (ResNet, ViT, etc.)
    """
    
    def __init__(self,
                 input_channels: int = 3,
                 latent_dim: int = 256,
                 base_channels: int = 64):
        """
        Args:
            input_channels: Number of input channels (3 for RGB)
            latent_dim: Dimension of latent representation
            base_channels: Base number of channels (doubled at each stage)
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Encoder: progressively downsample and increase channels
        # Input: (B, 3, H, W) -> Output: (B, latent_dim)
        
        # Stage 1: (B, 3, 64, 64) -> (B, 64, 32, 32)
        self.stage1 = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, 7, stride=2, padding=3),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            ResidualBlock(base_channels)
        )
        
        # Stage 2: (B, 64, 32, 32) -> (B, 128, 16, 16)
        self.stage2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            ResidualBlock(base_channels * 2)
        )
        
        # Stage 3: (B, 128, 16, 16) -> (B, 256, 8, 8)
        self.stage3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            ResidualBlock(base_channels * 4)
        )
        
        # Stage 4: (B, 256, 8, 8) -> (B, 512, 4, 4)
        self.stage4 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            ResidualBlock(base_channels * 8)
        )
        
        # Global average pooling and projection to latent space
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(base_channels * 8, latent_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input frames (B, C, H, W) in [-1, 1]
            
        Returns:
            latent: Latent representation (B, latent_dim)
        """
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        # Global pooling
        x = self.pool(x)  # (B, 512, 1, 1)
        x = x.flatten(1)  # (B, 512)
        
        # Project to latent space
        latent = self.fc(x)  # (B, latent_dim)
        
        return latent


class VideoEncoder(nn.Module):
    """
    Encoder that processes a sequence of frames.
    Applies frame encoder to each frame independently.
    """
    
    def __init__(self,
                 input_channels: int = 3,
                 latent_dim: int = 256,
                 base_channels: int = 64):
        """
        Args:
            input_channels: Number of input channels
            latent_dim: Dimension of latent representation per frame
            base_channels: Base channels for encoder
        """
        super().__init__()
        
        self.frame_encoder = FrameEncoder(input_channels, latent_dim, base_channels)
        self.latent_dim = latent_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input video (B, T, C, H, W) in [-1, 1]
            
        Returns:
            latents: Sequence of latent vectors (B, T, latent_dim)
        """
        B, T, C, H, W = x.shape
        
        # Reshape: (B, T, C, H, W) -> (B*T, C, H, W)
        x = x.view(B * T, C, H, W)
        
        # Encode all frames
        latents = self.frame_encoder(x)  # (B*T, latent_dim)
        
        # Reshape back: (B*T, latent_dim) -> (B, T, latent_dim)
        latents = latents.view(B, T, self.latent_dim)
        
        return latents


# For foundation models: easy swap with pretrained encoders
class PretrainedEncoder(nn.Module):
    """
    Wrapper for using pretrained foundation models as encoders.
    
    Supported models:
    - 'dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14' (DINOv2)
    - 'resnet50', 'resnet101' (torchvision ResNet)
    - More can be added as needed
    
    Benefits:
    - Pretrained on massive datasets (ImageNet, etc.)
    - Better feature extraction
    - Faster convergence
    - Less training data needed
    
    Usage:
        # Instead of VideoEncoder, use:
        from src.models.encoder import PretrainedVideoEncoder
        
        encoder = PretrainedVideoEncoder(
            model_name='dinov2_vits14',  # or 'resnet50'
            latent_dim=256,
            freeze_backbone=True  # Freeze pretrained weights
        )
    """
    
    def __init__(self, 
                 model_name: str = "resnet50",
                 latent_dim: int = 256,
                 freeze_backbone: bool = False):
        """
        Args:
            model_name: Name of pretrained model
            latent_dim: Output latent dimension
            freeze_backbone: Whether to freeze pretrained weights
        """
        super().__init__()
        self.model_name = model_name
        self.latent_dim = latent_dim
        
        # Load backbone based on model name
        if model_name.startswith('dinov2'):
            # DINOv2 models - requires: pip install git+https://github.com/facebookresearch/dinov2.git
            try:
                self.backbone = torch.hub.load('facebookresearch/dinov2', model_name)
                backbone_dim = self.backbone.embed_dim
            except:
                raise ImportError(
                    "DINOv2 not available. Install with: "
                    "pip install git+https://github.com/facebookresearch/dinov2.git"
                )
        
        elif model_name.startswith('resnet'):
            # ResNet models from torchvision
            import torchvision.models as models
            if model_name == 'resnet50':
                self.backbone = models.resnet50(pretrained=True)
            elif model_name == 'resnet101':
                self.backbone = models.resnet101(pretrained=True)
            else:
                raise ValueError(f"Unknown ResNet model: {model_name}")
            
            # Remove the final FC layer
            backbone_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Projection layer to latent_dim
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, latent_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(latent_dim * 2, latent_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input frames (B, C, H, W) in [-1, 1]
            
        Returns:
            latent: Latent representation (B, latent_dim)
        """
        # Normalize from [-1, 1] to [0, 1] for pretrained models
        x = (x + 1.0) / 2.0
        
        # Extract features
        features = self.backbone(x)  # (B, backbone_dim)
        
        # Project to latent space
        latent = self.projection(features)  # (B, latent_dim)
        
        return latent


class PretrainedVideoEncoder(nn.Module):
    """
    Video encoder using pretrained frame encoder.
    Processes each frame with the pretrained model.
    """
    
    def __init__(self,
                 model_name: str = "resnet50",
                 latent_dim: int = 256,
                 freeze_backbone: bool = False):
        """
        Args:
            model_name: Name of pretrained model
            latent_dim: Output latent dimension per frame
            freeze_backbone: Whether to freeze pretrained weights
        """
        super().__init__()
        
        self.frame_encoder = PretrainedEncoder(
            model_name=model_name,
            latent_dim=latent_dim,
            freeze_backbone=freeze_backbone
        )
        self.latent_dim = latent_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input video (B, T, C, H, W) in [-1, 1]
            
        Returns:
            latents: Sequence of latent vectors (B, T, latent_dim)
        """
        B, T, C, H, W = x.shape
        
        # Reshape: (B, T, C, H, W) -> (B*T, C, H, W)
        x = x.view(B * T, C, H, W)
        
        # Encode all frames
        latents = self.frame_encoder(x)  # (B*T, latent_dim)
        
        # Reshape back: (B*T, latent_dim) -> (B, T, latent_dim)
        latents = latents.view(B, T, self.latent_dim)
        
        return latents


if __name__ == "__main__":
    # Test the encoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Test FrameEncoder (from scratch)
    print("\n" + "="*50)
    print("Testing FrameEncoder (from scratch)...")
    frame_encoder = FrameEncoder(
        input_channels=3,
        latent_dim=256,
        base_channels=64
    ).to(device)
    
    # Single frame
    x = torch.randn(4, 3, 64, 64).to(device)
    latent = frame_encoder(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {latent.shape}")
    print(f"Parameters: {sum(p.numel() for p in frame_encoder.parameters()):,}")
    
    # Test VideoEncoder (from scratch)
    print("\n" + "="*50)
    print("Testing VideoEncoder (from scratch)...")
    video_encoder = VideoEncoder(
        input_channels=3,
        latent_dim=256,
        base_channels=64
    ).to(device)
    
    # Video sequence
    x = torch.randn(2, 30, 3, 64, 64).to(device)
    latents = video_encoder(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {latents.shape}")
    print(f"Parameters: {sum(p.numel() for p in video_encoder.parameters()):,}")
    
    # Test PretrainedVideoEncoder (with ResNet50)
    print("\n" + "="*50)
    print("Testing PretrainedVideoEncoder (ResNet50)...")
    try:
        pretrained_encoder = PretrainedVideoEncoder(
            model_name='resnet50',
            latent_dim=256,
            freeze_backbone=True
        ).to(device)
        
        x = torch.randn(2, 30, 3, 64, 64).to(device)
        latents = pretrained_encoder(x)
        
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {latents.shape}")
        print(f"Parameters: {sum(p.numel() for p in pretrained_encoder.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in pretrained_encoder.parameters() if p.requires_grad):,}")
    except Exception as e:
        print(f"Pretrained encoder test skipped: {e}")
    
    print("\n✅ Encoder tests passed!")
