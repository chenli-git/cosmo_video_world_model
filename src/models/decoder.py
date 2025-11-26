"""
Decoder: reconstructs frames from latent representations.
Includes both simple CNN decoder and advanced decoder with attention.
"""
import torch
import torch.nn as nn
from typing import Optional


class ResidualBlock(nn.Module):
    """Residual block for decoder."""
    
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


class FrameDecoder(nn.Module):
    """
    Decoder that reconstructs a frame from a latent vector.
    
    Architecture: Latent vector -> transposed conv layers -> output frame
    """
    
    def __init__(self,
                 latent_dim: int = 256,
                 output_channels: int = 3,
                 base_channels: int = 64,
                 output_size: int = 64):
        """
        Args:
            latent_dim: Dimension of input latent vector
            output_channels: Number of output channels (3 for RGB)
            base_channels: Base number of channels
            output_size: Output frame size (assumes square)
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.output_size = output_size
        
        # Project latent to spatial features
        # Start with 4x4 spatial resolution
        self.fc = nn.Linear(latent_dim, base_channels * 8 * 4 * 4)
        self.initial_size = 4
        
        # Decoder stages: progressively upsample
        # (B, 512, 4, 4) -> (B, 256, 8, 8) -> (B, 128, 16, 16) -> (B, 64, 32, 32) -> (B, 3, 64, 64)
        
        # Stage 1: (B, 512, 4, 4) -> (B, 256, 8, 8)
        self.stage1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            ResidualBlock(base_channels * 4)
        )
        
        # Stage 2: (B, 256, 8, 8) -> (B, 128, 16, 16)
        self.stage2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            ResidualBlock(base_channels * 2)
        )
        
        # Stage 3: (B, 128, 16, 16) -> (B, 64, 32, 32)
        self.stage3 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            ResidualBlock(base_channels)
        )
        
        # Stage 4: (B, 64, 32, 32) -> (B, 3, 64, 64)
        self.stage4 = nn.Sequential(
            nn.ConvTranspose2d(base_channels, output_channels, 4, stride=2, padding=1),
            nn.Tanh()  # Output in [-1, 1] to match input normalization
        )
    
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: Latent vector (B, latent_dim)
            
        Returns:
            frame: Reconstructed frame (B, C, H, W) in [-1, 1]
        """
        # Project to spatial features
        x = self.fc(latent)  # (B, 512*4*4)
        x = x.view(-1, 512, self.initial_size, self.initial_size)  # (B, 512, 4, 4)
        
        # Upsample progressively
        x = self.stage1(x)  # (B, 256, 8, 8)
        x = self.stage2(x)  # (B, 128, 16, 16)
        x = self.stage3(x)  # (B, 64, 32, 32)
        x = self.stage4(x)  # (B, 3, 64, 64)
        
        return x


class SelfAttention(nn.Module):
    """Self-attention module for decoder."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Compute Q, K, V
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)  # (B, HW, C//8)
        k = self.key(x).view(B, -1, H * W)  # (B, C//8, HW)
        v = self.value(x).view(B, -1, H * W)  # (B, C, HW)
        
        # Attention
        attention = torch.bmm(q, k)  # (B, HW, HW)
        attention = torch.softmax(attention, dim=-1)
        
        # Apply attention to values
        out = torch.bmm(v, attention.permute(0, 2, 1))  # (B, C, HW)
        out = out.view(B, C, H, W)
        
        # Residual connection
        out = self.gamma * out + x
        return out


class EnhancedFrameDecoder(nn.Module):
    """
    Enhanced decoder with attention and skip connections.
    Better for use with powerful encoders like DINOv2.
    
    Features:
    - Deeper architecture
    - Self-attention at multiple scales
    - More residual blocks
    - Better detail reconstruction
    """
    
    def __init__(self,
                 latent_dim: int = 256,
                 output_channels: int = 3,
                 base_channels: int = 64,
                 output_size: int = 64,
                 use_attention: bool = True):
        """
        Args:
            latent_dim: Dimension of input latent vector
            output_channels: Number of output channels (3 for RGB)
            base_channels: Base number of channels
            output_size: Output frame size (assumes square)
            use_attention: Whether to use self-attention layers
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.output_size = output_size
        self.use_attention = use_attention
        
        # Project latent to spatial features with larger initial size
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim * 2, base_channels * 8 * 4 * 4)
        )
        self.initial_size = 4
        
        # Stage 1: (B, 512, 4, 4) -> (B, 256, 8, 8)
        self.stage1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            ResidualBlock(base_channels * 4),
            ResidualBlock(base_channels * 4)
        )
        if use_attention:
            self.attn1 = SelfAttention(base_channels * 4)
        
        # Stage 2: (B, 256, 8, 8) -> (B, 128, 16, 16)
        self.stage2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            ResidualBlock(base_channels * 2),
            ResidualBlock(base_channels * 2)
        )
        if use_attention:
            self.attn2 = SelfAttention(base_channels * 2)
        
        # Stage 3: (B, 128, 16, 16) -> (B, 64, 32, 32)
        self.stage3 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            ResidualBlock(base_channels),
            ResidualBlock(base_channels)
        )
        
        # Stage 4: (B, 64, 32, 32) -> (B, 3, 64, 64)
        self.stage4 = nn.Sequential(
            nn.ConvTranspose2d(base_channels, base_channels // 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels // 2, output_channels, 3, padding=1),
            nn.Tanh()
        )
    
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: Latent vector (B, latent_dim)
            
        Returns:
            frame: Reconstructed frame (B, C, H, W) in [-1, 1]
        """
        # Project to spatial features
        x = self.fc(latent)
        x = x.view(-1, 512, self.initial_size, self.initial_size)
        
        # Upsample with attention
        x = self.stage1(x)
        if self.use_attention:
            x = self.attn1(x)
        
        x = self.stage2(x)
        if self.use_attention:
            x = self.attn2(x)
        
        x = self.stage3(x)
        x = self.stage4(x)
        
        return x


class VideoDecoder(nn.Module):
    """
    Decoder that reconstructs a sequence of frames from latent representations.
    Applies frame decoder to each latent independently.
    """
    
    def __init__(self,
                 latent_dim: int = 256,
                 output_channels: int = 3,
                 base_channels: int = 64,
                 output_size: int = 64,
                 use_enhanced: bool = False):
        """
        Args:
            latent_dim: Dimension of latent vectors
            output_channels: Number of output channels
            base_channels: Base channels for decoder
            output_size: Output frame size
            use_enhanced: Use enhanced decoder with attention (recommended for DINOv2/strong encoders)
        """
        super().__init__()
        
        if use_enhanced:
            self.frame_decoder = EnhancedFrameDecoder(
                latent_dim, output_channels, base_channels, output_size, use_attention=True
            )
        else:
            self.frame_decoder = FrameDecoder(
                latent_dim, output_channels, base_channels, output_size
            )
        
        self.latent_dim = latent_dim
        self.use_enhanced = use_enhanced
    
    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latents: Sequence of latent vectors (B, T, latent_dim)
            
        Returns:
            frames: Reconstructed frames (B, T, C, H, W) in [-1, 1]
        """
        B, T, D = latents.shape
        
        # Reshape: (B, T, D) -> (B*T, D)
        latents = latents.view(B * T, D)
        
        # Decode all frames
        frames = self.frame_decoder(latents)  # (B*T, C, H, W)
        
        # Reshape back: (B*T, C, H, W) -> (B, T, C, H, W)
        _, C, H, W = frames.shape
        frames = frames.view(B, T, C, H, W)
        
        return frames


if __name__ == "__main__":
    # Test the decoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Test FrameDecoder
    print("\n" + "="*50)
    print("Testing FrameDecoder...")
    frame_decoder = FrameDecoder(
        latent_dim=256,
        output_channels=3,
        base_channels=64,
        output_size=64
    ).to(device)
    
    # Single latent
    latent = torch.randn(4, 256).to(device)
    frame = frame_decoder(latent)
    
    print(f"Input shape: {latent.shape}")
    print(f"Output shape: {frame.shape}")
    print(f"Output range: [{frame.min():.2f}, {frame.max():.2f}]")
    print(f"Parameters: {sum(p.numel() for p in frame_decoder.parameters()):,}")
    
    # Test VideoDecoder (simple)
    print("\n" + "="*50)
    print("Testing VideoDecoder (simple)...")
    video_decoder = VideoDecoder(
        latent_dim=256,
        output_channels=3,
        base_channels=64,
        output_size=64,
        use_enhanced=False
    ).to(device)
    
    # Sequence of latents
    latents = torch.randn(2, 30, 256).to(device)
    frames = video_decoder(latents)
    
    print(f"Input shape: {latents.shape}")
    print(f"Output shape: {frames.shape}")
    print(f"Output range: [{frames.min():.2f}, {frames.max():.2f}]")
    print(f"Parameters: {sum(p.numel() for p in video_decoder.parameters()):,}")
    
    # Test EnhancedVideoDecoder
    print("\n" + "="*50)
    print("Testing VideoDecoder (enhanced with attention)...")
    enhanced_decoder = VideoDecoder(
        latent_dim=256,
        output_channels=3,
        base_channels=64,
        output_size=64,
        use_enhanced=True
    ).to(device)
    
    frames = enhanced_decoder(latents)
    
    print(f"Input shape: {latents.shape}")
    print(f"Output shape: {frames.shape}")
    print(f"Output range: [{frames.min():.2f}, {frames.max():.2f}]")
    print(f"Parameters: {sum(p.numel() for p in enhanced_decoder.parameters()):,}")
    
    print("\n✅ Decoder tests passed!")
