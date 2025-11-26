# Video World Model: Learning Physics from Real Videos

A lightweight implementation inspired by Cosmos that learns to predict future video frames by understanding physics from real-world videos.

## 🎯 Project Overview

This project implements a video world model that:
- Learns physics dynamics from synthetic videos (bouncing balls, pendulums, falling objects, projectiles)
- Predicts 30 future frames given 30 input frames
- Uses CNN encoder → latent dynamics (GRU/Transformer) → decoder architecture

## 🏗️ Architecture

```
Input Video Frames
      ↓
Encoder (Custom CNN or Pretrained: ResNet/DINOv2)
      ↓
Latent Representation (256-dim compressed)
      ↓
Dynamics Model (GRU/LSTM/Transformer)
      ↓
Future Latent States
      ↓
Decoder (Simple CNN or Enhanced with Attention)
      ↓
Predicted Future Frames
```

### Key Features:
- **Pretrained Encoder Support**: Use foundation models (ResNet50/101, DINOv2) for better features
- **Enhanced Decoder**: Self-attention layers for high-quality reconstruction with powerful encoders
- **Flexible Dynamics**: Choose GRU (fast), LSTM, or Transformer (better long-term)
- **Automatic Architecture Matching**: Enhanced decoder activated automatically with pretrained encoders

## 📁 Project Structure

```
.
├── data/                      # Training videos and datasets
│   ├── raw/                   # Raw video files (512 videos)
│   └── split/                 # Train/val/test split
│       ├── train/             # Training videos (409 videos)
│       ├── val/               # Validation videos (51 videos)
│       └── test/              # Test videos (52 videos)
├── src/                       # Source code
│   ├── models/                # Model architectures
│   │   ├── encoder.py         # CNN encoder + Pretrained (ResNet/DINOv2)
│   │   ├── decoder.py         # CNN decoder + Enhanced (with attention)
│   │   ├── world_model.py     # GRU/LSTM/Transformer dynamics
│   │   ├── video_world_model.py  # Complete end-to-end model
│   │   └── __init__.py        # Model exports
│   └── data/                  # Data loading and preprocessing
│       ├── dataset.py         # PyTorch VideoClipDataset
│       └── __init__.py        # Data exports
├── tools/                     # Utility scripts
│   ├── generate_sample_videos.py  # Create synthetic physics videos
│   ├── split_dataset.py       # Split videos into train/val/test
│   ├── train_model.py         # Main training script (TODO)
│   └── inference.py           # Run inference on videos (TODO)
├── notebooks/                 # Jupyter notebooks for demos
│   └── demo.ipynb             # Interactive demo
├── checkpoints/               # Saved model checkpoints
└── outputs/                   # Generated predictions
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
conda activate cosmos-lite
pip install -r requirements.txt
```

### 2. Generate Sample Videos

```bash
python tools/generate_sample_videos.py --output data/raw --num-videos 128
```

This creates 512 synthetic physics videos (128 per type):
- **Bouncing balls** (1-5 balls with realistic collisions) - 256x256 resolution
- **Pendulums** (simple harmonic motion) - 64x64 resolution
- **Falling objects** (gravity + air resistance) - 64x64 resolution
- **Projectiles** (parabolic trajectories with trails) - 64x64 resolution

Each video has 60 frames at 30 FPS.

### 3. Split Dataset into Train/Val/Test

```bash
python tools/split_dataset.py --input data/raw --output data/split
```

This splits the 512 videos into:
- **Train**: 409 videos (80%)
- **Val**: 51 videos (10%)
- **Test**: 52 videos (10%)

### 4. Train the Model

#### Basic Usage (Custom Encoder)
```python
from src.models import VideoWorldModel

# Create model with custom encoder
model = VideoWorldModel(
    input_channels=3,
    latent_dim=256,
    hidden_dim=512,
    frame_size=64,
    world_model_type='gru',  # or 'lstm', 'transformer'
    use_pretrained_encoder=False
)
```

#### Advanced Usage (Pretrained Encoder)
```python
# Use DINOv2 for better quality
model = VideoWorldModel(
    input_channels=3,
    latent_dim=256,
    hidden_dim=512,
    frame_size=64,
    world_model_type='transformer',
    use_pretrained_encoder=True,
    pretrained_model_name='dinov2_vits14',  # or 'resnet50', 'resnet101', 'dinov2_vitb14'
    freeze_encoder=False  # Set True to freeze pretrained weights
)
# Enhanced decoder automatically activated!
```

### 5. Load and Use the Dataset

```python
from src.data import create_data_loader

# Create train loader
train_loader = create_data_loader(
    video_dir='data/split/train',
    T_in=30,        # 30 input frames
    T_out=30,       # Predict 30 future frames
    batch_size=4,
    stride=5,
    resize=(64, 64),
    shuffle=True,
    augment=True    # Data augmentation for training
)

# Create val loader
val_loader = create_data_loader(
    video_dir='data/split/val',
    T_in=30,
    T_out=30,
    batch_size=4,
    shuffle=False,
    augment=False
)

# Iterate through batches
for context, target in train_loader:
    # context: (B, T_in, C, H, W) - input frames
    # target: (B, T_out, C, H, W) - frames to predict
    pass
```

### 5. Run Inference

```bash
python tools/inference.py \
    --checkpoint checkpoints/best_model.pt \
    --input path/to/video.mp4 \
    --output outputs/prediction.mp4
```

## 🎮 Interactive Demo

Check out `notebooks/demo.ipynb` for an interactive demonstration of:
- Loading and visualizing training data
- Training a small model
- Generating predictions
- Comparing predictions with ground truth

## 📊 Model Details

### Encoder Options

**1. Custom CNN Encoder** (default)
- ResNet-style architecture with residual blocks
- Compresses 64×64×3 frames to 256-dim latent vectors
- ~6.9M parameters
- Trains from scratch on your data

**2. Pretrained Encoder** (recommended for better quality)
- **ResNet50/101**: ImageNet pretrained, excellent general features
- **DINOv2 (ViT-S/B/L/G)**: Self-supervised vision transformer, state-of-the-art features
- Projects pretrained features to 256-dim latents
- Optional backbone freezing for faster training
- Better generalization with less data

### Dynamics Model
Three options:
1. **GRU** (default): ~3.2M params, fast training, good for short-term
2. **LSTM**: ~4.2M params, alternative RNN with better memory
3. **Transformer**: ~7.4M params, attention-based, best for long-term predictions

### Decoder Options

**1. Simple Decoder** (default)
- Transposed CNN with residual blocks
- ~3.6M parameters
- Good for custom encoder

**2. Enhanced Decoder** (auto-enabled with pretrained encoders)
- Deeper architecture with self-attention layers
- ~5.8M parameters
- Better detail reconstruction for powerful encoders like DINOv2
- Attention at multiple spatial scales (8×8, 16×16)

### Complete Model Sizes
- **Custom CNN + GRU**: ~17.5M parameters
- **Custom CNN + Transformer**: ~21.8M parameters
- **DINOv2 + GRU + Enhanced Decoder**: Variable (DINOv2-S: ~30M total)

### Loss Functions
- **Reconstruction Loss**: MSE or L1 between predicted and actual frames
- **Perceptual Loss**: VGG-based feature matching
- **Temporal Consistency**: Smooth motion between frames

## 🎯 Why This is "Cosmos"

This project captures the essence of Cosmos:
1. **Physical Understanding**: Learns implicit physics from raw video
2. **World Model**: Predicts future states based on learned dynamics
3. **Generalization**: Works across different physics scenarios
4. **Scalability**: Architecture scales from simple to complex scenes

## 📈 Dataset Statistics

- **Total videos**: 512 (60 frames each, 30 FPS)
- **Train/Val/Test split**: 80%/10%/10% (409/51/52 videos)
- **Clips per video**: ~5 clips (with stride=5)
- **Total training clips**: ~23,000 clips
- **Clip format**: 30 input frames → predict 30 output frames (60 frames total)
- **Resolution**: 
  - Bouncing balls: 256×256 
  - Other types: 64×64

## 📈 Training Tips

- Start with simple scenarios (single bouncing ball)
- Input frames (T_in): 30, Output frames (T_out): 30
- Learning rate: 1e-4 with cosine annealing
- Batch size: 4-8 depending on GPU memory
- Use data augmentation (horizontal flip, brightness) for training

## 🔧 Customization

### Choose Your Encoder

**For Quick Start / Limited Data:**
- Use pretrained encoders (ResNet or DINOv2)
- Freeze backbone initially: `freeze_encoder=True`
- Fine-tune later for better results

**For Custom Physics / Abundant Data:**
- Train custom encoder from scratch
- Full control over feature learning
- Smaller model size

### Use Your Own Videos

1. Place videos in `data/raw/`
2. Run split script: `python tools/split_dataset.py`
3. Train with your custom data

### Modify Architecture

Edit files in `src/models/` to experiment with:
- Different encoder backbones (add new pretrained models)
- Custom decoder architectures
- Attention mechanisms (spatial, temporal, cross-attention)
- Diffusion-based prediction
- Multi-scale processing

## 📝 Example Results

After training, you should see:
- Accurate short-term predictions (1-10 frames)
- Reasonable physics (objects don't teleport)
- Smooth motion trajectories
- Gradual quality degradation for long-term predictions

## 🙏 Acknowledgments

Inspired by NVIDIA Cosmos and world models research.

## 📄 License

MIT License - feel free to use for learning and portfolio projects!
