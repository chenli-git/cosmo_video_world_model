"""
Complete Video World Model: combines encoder, world model, and decoder.
"""
import torch
import torch.nn as nn
from typing import Literal, Tuple, Optional

from .encoder import VideoEncoder, PretrainedVideoEncoder
from .decoder import VideoDecoder
from .world_model import GRUWorldModel, TransformerWorldModel, LSTMWorldModel


class VideoWorldModel(nn.Module):
    """
    Complete video world model for physics prediction.
    
    Architecture:
        Input frames -> Encoder -> Latent sequence
                                        ↓
        Predicted frames <- Decoder <- World Model (predict future latents)
    """
    
    def __init__(self,
                 # Encoder params
                 input_channels: int = 3,
                 latent_dim: int = 256,
                 base_channels: int = 64,
                 # Pretrained encoder (optional)
                 use_pretrained_encoder: bool = False,
                 pretrained_model_name: Optional[str] = None,
                 freeze_encoder: bool = False,
                 # World model params
                 world_model_type: Literal['gru', 'lstm', 'transformer'] = 'gru',
                 hidden_dim: int = 512,
                 num_layers: int = 2,
                 num_heads: int = 8,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 # Decoder params
                 output_size: int = 64):
        """
        Args:
            input_channels: Number of input channels (3 for RGB)
            latent_dim: Dimension of latent space
            base_channels: Base channels for encoder/decoder
            use_pretrained_encoder: Use pretrained foundation model for encoder
            pretrained_model_name: Name of pretrained model ('resnet50', 'dinov2_vits14', etc.)
            freeze_encoder: Freeze encoder weights (for pretrained or after pretraining)
            world_model_type: Type of world model ('gru', 'lstm', 'transformer')
            hidden_dim: Hidden dimension for RNN models
            num_layers: Number of layers
            num_heads: Number of attention heads (for transformer)
            dim_feedforward: Feedforward dimension (for transformer)
            dropout: Dropout probability
            output_size: Output frame size
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.world_model_type = world_model_type
        self.use_pretrained_encoder = use_pretrained_encoder
        
        # Encoder: frames -> latents
        if use_pretrained_encoder:
            if pretrained_model_name is None:
                pretrained_model_name = 'resnet50'
            
            print(f"Using pretrained encoder: {pretrained_model_name}")
            self.encoder = PretrainedVideoEncoder(
                model_name=pretrained_model_name,
                latent_dim=latent_dim,
                freeze_backbone=freeze_encoder
            )
        else:
            self.encoder = VideoEncoder(
                input_channels=input_channels,
                latent_dim=latent_dim,
                base_channels=base_channels
            )
            
            # Optionally freeze encoder (useful after pretraining)
            if freeze_encoder:
                for param in self.encoder.parameters():
                    param.requires_grad = False
        
        # World model: context latents -> future latents
        if world_model_type == 'gru':
            self.world_model = GRUWorldModel(
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout
            )
        elif world_model_type == 'lstm':
            self.world_model = LSTMWorldModel(
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout
            )
        elif world_model_type == 'transformer':
            self.world_model = TransformerWorldModel(
                latent_dim=latent_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unknown world model type: {world_model_type}")
        
        # Decoder: latents -> frames
        # Use enhanced decoder with attention when using pretrained encoders for better quality
        self.decoder = VideoDecoder(
            latent_dim=latent_dim,
            output_channels=input_channels,
            base_channels=base_channels,
            output_size=output_size,
            use_enhanced=use_pretrained_encoder
        )
    
    def forward(self, context_frames: torch.Tensor,
                num_predictions: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass: predict future frames from context frames.
        
        Args:
            context_frames: Input context frames (B, T_in, C, H, W) in [-1, 1]
            num_predictions: Number of future frames to predict (T_out)
            
        Returns:
            predicted_frames: Predicted frames (B, T_out, C, H, W) in [-1, 1]
            context_latents: Encoded context latents (B, T_in, latent_dim)
            predicted_latents: Predicted future latents (B, T_out, latent_dim)
        """
        # Encode context frames to latent space
        context_latents = self.encoder(context_frames)  # (B, T_in, latent_dim)
        
        # Predict future latents
        predicted_latents = self.world_model(
            context_latents, num_predictions
        )  # (B, T_out, latent_dim)
        
        # Decode predicted latents to frames
        predicted_frames = self.decoder(predicted_latents)  # (B, T_out, C, H, W)
        
        return predicted_frames, context_latents, predicted_latents
    
    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode frames to latent space."""
        return self.encoder(frames)
    
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to frames."""
        return self.decoder(latents)
    
    def reconstruct(self, frames: torch.Tensor) -> torch.Tensor:
        """Reconstruct frames (encode then decode)."""
        latents = self.encode(frames)
        return self.decode(latents)
    
    def predict_recursive(self, context_frames: torch.Tensor,
                         num_predictions: int) -> torch.Tensor:
        """
        Predict frames recursively using previously predicted frames as context.
        This can generate longer sequences but may accumulate errors.
        
        Args:
            context_frames: Initial context frames (B, T_in, C, H, W)
            num_predictions: Number of frames to predict
            
        Returns:
            predicted_frames: All predicted frames (B, num_predictions, C, H, W)
        """
        all_predictions = []
        current_context = context_frames
        
        # Predict in chunks
        while len(all_predictions) < num_predictions:
            # Predict next chunk
            chunk_size = min(num_predictions - len(all_predictions), 
                           current_context.shape[1])
            
            predicted, _, _ = self.forward(current_context, chunk_size)
            all_predictions.append(predicted)
            
            # Update context for next iteration
            current_context = torch.cat([
                current_context[:, chunk_size:],
                predicted
            ], dim=1)
            
            if current_context.shape[1] == 0:
                break
        
        return torch.cat(all_predictions, dim=1)
    
    def get_num_parameters(self) -> dict:
        """Get parameter counts for each component."""
        return {
            'encoder': sum(p.numel() for p in self.encoder.parameters()),
            'world_model': sum(p.numel() for p in self.world_model.parameters()),
            'decoder': sum(p.numel() for p in self.decoder.parameters()),
            'total': sum(p.numel() for p in self.parameters())
        }


if __name__ == "__main__":
    # Test the complete model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    B, T_in, T_out = 2, 30, 30
    C, H, W = 3, 64, 64
    
    # Test with GRU
    print("\n" + "="*50)
    print("Testing VideoWorldModel with GRU...")
    model_gru = VideoWorldModel(
        input_channels=3,
        latent_dim=256,
        base_channels=64,
        world_model_type='gru',
        hidden_dim=512,
        num_layers=2,
        output_size=64
    ).to(device)
    
    context = torch.randn(B, T_in, C, H, W).to(device)
    predicted, context_latents, predicted_latents = model_gru(context, T_out)
    
    print(f"Context frames: {context.shape}")
    print(f"Predicted frames: {predicted.shape}")
    print(f"Context latents: {context_latents.shape}")
    print(f"Predicted latents: {predicted_latents.shape}")
    
    params = model_gru.get_num_parameters()
    print(f"\nParameter counts:")
    for name, count in params.items():
        print(f"  {name}: {count:,}")
    
    # Test reconstruction
    print("\n" + "="*50)
    print("Testing reconstruction...")
    reconstructed = model_gru.reconstruct(context)
    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"Reconstruction error (MSE): {torch.mean((context - reconstructed) ** 2).item():.6f}")
    
    # Test with Transformer
    print("\n" + "="*50)
    print("Testing VideoWorldModel with Transformer...")
    model_transformer = VideoWorldModel(
        input_channels=3,
        latent_dim=256,
        base_channels=64,
        world_model_type='transformer',
        num_heads=8,
        num_layers=4,
        output_size=64
    ).to(device)
    
    predicted, _, _ = model_transformer(context, T_out)
    print(f"Predicted frames: {predicted.shape}")
    
    params = model_transformer.get_num_parameters()
    print(f"\nParameter counts:")
    for name, count in params.items():
        print(f"  {name}: {count:,}")
    
    print("\n✅ All tests passed!")
