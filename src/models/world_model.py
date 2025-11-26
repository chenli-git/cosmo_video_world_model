"""
World Model (Dynamics): predicts future latent states.
Supports GRU and Transformer architectures.
"""
import torch
import torch.nn as nn
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D)
        """
        x = x + self.pe[:, :x.size(1), :]
        return x


class GRUWorldModel(nn.Module):
    """
    GRU-based world model for dynamics prediction.
    Fast and efficient for shorter sequences.
    """
    
    def __init__(self,
                 latent_dim: int = 256,
                 hidden_dim: int = 512,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        """
        Args:
            latent_dim: Dimension of latent vectors
            hidden_dim: Hidden dimension of GRU
            num_layers: Number of GRU layers
            dropout: Dropout probability
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # GRU for context processing
        self.gru = nn.GRU(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim)
        )
    
    def forward(self, context_latents: torch.Tensor, 
                num_predictions: int) -> torch.Tensor:
        """
        Args:
            context_latents: Context latent sequence (B, T_in, latent_dim)
            num_predictions: Number of future steps to predict (T_out)
            
        Returns:
            predicted_latents: Predicted latent sequence (B, T_out, latent_dim)
        """
        B = context_latents.shape[0]
        
        # Process context through GRU
        _, hidden = self.gru(context_latents)  # hidden: (num_layers, B, hidden_dim)
        
        # Autoregressive prediction
        predictions = []
        current_input = context_latents[:, -1:, :]  # Last context frame
        
        for _ in range(num_predictions):
            # Predict next state
            output, hidden = self.gru(current_input, hidden)
            next_latent = self.output_proj(output)  # (B, 1, latent_dim)
            
            predictions.append(next_latent)
            current_input = next_latent
        
        # Concatenate predictions
        predicted_latents = torch.cat(predictions, dim=1)  # (B, T_out, latent_dim)
        
        return predicted_latents


class TransformerWorldModel(nn.Module):
    """
    Transformer-based world model for dynamics prediction.
    Better for longer sequences and complex dependencies.
    """
    
    def __init__(self,
                 latent_dim: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1):
        """
        Args:
            latent_dim: Dimension of latent vectors
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout probability
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(latent_dim)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=latent_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Linear(latent_dim, latent_dim)
    
    def forward(self, context_latents: torch.Tensor,
                num_predictions: int) -> torch.Tensor:
        """
        Args:
            context_latents: Context latent sequence (B, T_in, latent_dim)
            num_predictions: Number of future steps to predict (T_out)
            
        Returns:
            predicted_latents: Predicted latent sequence (B, T_out, latent_dim)
        """
        B, T_in, D = context_latents.shape
        device = context_latents.device
        
        # Add positional encoding to context
        context_encoded = self.pos_encoder(context_latents)
        
        # Autoregressive generation
        predictions = []
        
        # Start with the last context frame
        current_sequence = context_latents[:, -1:, :]
        
        for step in range(num_predictions):
            # Add positional encoding
            current_encoded = self.pos_encoder(current_sequence)
            
            # Generate causal mask for decoder
            tgt_len = current_sequence.shape[1]
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                tgt_len, device=device
            )
            
            # Predict next frame
            output = self.transformer(
                src=context_encoded,
                tgt=current_encoded,
                tgt_mask=tgt_mask
            )
            
            # Get prediction for next frame
            next_latent = self.output_proj(output[:, -1:, :])  # (B, 1, latent_dim)
            
            predictions.append(next_latent)
            current_sequence = torch.cat([current_sequence, next_latent], dim=1)
        
        # Concatenate all predictions
        predicted_latents = torch.cat(predictions, dim=1)  # (B, T_out, latent_dim)
        
        return predicted_latents


class LSTMWorldModel(nn.Module):
    """
    LSTM-based world model (alternative to GRU).
    Similar performance but with more parameters.
    """
    
    def __init__(self,
                 latent_dim: int = 256,
                 hidden_dim: int = 512,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim)
        )
    
    def forward(self, context_latents: torch.Tensor,
                num_predictions: int) -> torch.Tensor:
        B = context_latents.shape[0]
        
        # Process context
        _, (hidden, cell) = self.lstm(context_latents)
        
        # Autoregressive prediction
        predictions = []
        current_input = context_latents[:, -1:, :]
        
        for _ in range(num_predictions):
            output, (hidden, cell) = self.lstm(current_input, (hidden, cell))
            next_latent = self.output_proj(output)
            
            predictions.append(next_latent)
            current_input = next_latent
        
        predicted_latents = torch.cat(predictions, dim=1)
        return predicted_latents


if __name__ == "__main__":
    # Test the world models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    B, T_in, T_out, latent_dim = 2, 30, 30, 256
    context = torch.randn(B, T_in, latent_dim).to(device)
    
    # Test GRU
    print("\n" + "="*50)
    print("Testing GRUWorldModel...")
    gru_model = GRUWorldModel(
        latent_dim=latent_dim,
        hidden_dim=512,
        num_layers=2
    ).to(device)
    
    predictions = gru_model(context, T_out)
    print(f"Context shape: {context.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Parameters: {sum(p.numel() for p in gru_model.parameters()):,}")
    
    # Test Transformer
    print("\n" + "="*50)
    print("Testing TransformerWorldModel...")
    transformer_model = TransformerWorldModel(
        latent_dim=latent_dim,
        num_heads=8,
        num_layers=4
    ).to(device)
    
    predictions = transformer_model(context, T_out)
    print(f"Context shape: {context.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Parameters: {sum(p.numel() for p in transformer_model.parameters()):,}")
    
    # Test LSTM
    print("\n" + "="*50)
    print("Testing LSTMWorldModel...")
    lstm_model = LSTMWorldModel(
        latent_dim=latent_dim,
        hidden_dim=512,
        num_layers=2
    ).to(device)
    
    predictions = lstm_model(context, T_out)
    print(f"Context shape: {context.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Parameters: {sum(p.numel() for p in lstm_model.parameters()):,}")
    
    print("\n✅ World model tests passed!")
