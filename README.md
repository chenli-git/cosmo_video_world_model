# Video World Model: Learning Physics from Real Videos

A lightweight implementation inspired by Cosmos that learns to predict future video frames by understanding physics from real-world videos.

## 🎯 Project Overview

This project implements a video world model that:
- **Learns physics dynamics** from synthetic videos (bouncing balls, pendulums, falling objects, projectiles)
- **Predicts future frames** autoregressively (30+ frames) given context frames
- **Supports pretrained encoders** (ResNet, DINOv2) for better feature extraction
- **Complete training pipeline** with loss functions, checkpointing, and TensorBoard logging
- **Visual demos** for autoregressive rollout with side-by-side comparisons

**Perfect for:** Learning about world models, physics prediction, video generation, or building a portfolio project!

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

### 1. Clone and Setup

```bash
git clone https://github.com/chenli-git/cosmo_video_world_model.git
cd cosmo_video_world_model

# Create conda environment
conda create -n cosmos-lite python=3.10
conda activate cosmos-lite

# Install dependencies
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

#### Option A: Quick Start (Custom Encoder, CPU-friendly)
```bash
python tools/train_model.py \
  --train-dir data/split/train \
  --val-dir data/split/val \
  --batch-size 4 \
  --epochs 100 \
  --world-model gru \
  --device cpu \
  --no-perceptual
```

#### Option B: Best Quality (Pretrained Encoder, GPU recommended)
```bash
python tools/train_model.py \
  --train-dir data/split/train \
  --val-dir data/split/val \
  --batch-size 8 \
  --epochs 100 \
  --world-model transformer \
  --use-pretrained \
  --pretrained-model dinov2_vits14 \
  --device cuda
```

**Monitor training:**
```bash
tensorboard --logdir runs
```
Open http://localhost:6006 in your browser.

### 5. Run Autoregressive Demo

Visualize predictions with side-by-side comparison:

```bash
python tools/rollout_demo.py \
  --checkpoint checkpoints/best_model.pt \
  --video-dir data/split/test \
  --T-in 30 \
  --num-predict 30 \
  --video-idx 0
```

Outputs:
- `outputs/demos/rollout_video_0.mp4` - Side-by-side comparison video
- `outputs/demos/rollout_grid_0.png` - Frame grid with MSE metrics

### 6. Programmatic Usage

#### Load Dataset
```python
from src.data import create_data_loader

# Create data loader
train_loader = create_data_loader(
    video_dir='data/split/train',
    T_in=30,
    T_out=30,
    batch_size=4,
    shuffle=True,
    augment=True
)
```

#### Create Model
```python
from src.models import VideoWorldModel

# Basic: Custom encoder
model = VideoWorldModel(
    input_channels=3,
    latent_dim=256,
    hidden_dim=512,
    frame_size=64,
    world_model_type='gru'
)

# Advanced: Pretrained encoder + enhanced decoder
model = VideoWorldModel(
    input_channels=3,
    latent_dim=256,
    hidden_dim=512,
    frame_size=64,
    world_model_type='transformer',
    use_pretrained_encoder=True,
    pretrained_model_name='dinov2_vits14',
    freeze_encoder=False
)
```

#### Train with Custom Loop

```python
from src.training import Trainer

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device='cuda'
)

trainer.train(num_epochs=100)
```

#### Inference
```python
import torch

model.eval()
with torch.no_grad():
    # Predict future frames
    pred_frames, pred_latents, context_latents = model(
        context_frames,  # (B, T_in, C, H, W)
        num_predictions=30
    )
    # pred_frames: (B, 30, C, H, W)
```

## 🛠️ Command-Line Tools

### Generate Dataset
```bash
# Generate synthetic physics videos
python tools/generate_sample_videos.py --output data/raw --num-videos 128

# Split into train/val/test
python tools/split_dataset.py --input data/raw --output data/split
```

### Train Model
```bash
# Basic training
python tools/train_model.py --train-dir data/split/train --val-dir data/split/val

# Advanced options
python tools/train_model.py \
  --train-dir data/split/train \
  --val-dir data/split/val \
  --batch-size 8 \
  --epochs 200 \
  --world-model transformer \
  --use-pretrained \
  --pretrained-model dinov2_vitb14 \
  --freeze-encoder \
  --lr 1e-4 \
  --recon-type l1 \
  --device cuda

# Resume training
python tools/train_model.py --resume checkpoints/checkpoint_epoch_50.pt
```

### Run Demos
```bash
# Autoregressive rollout demo
python tools/rollout_demo.py \
  --checkpoint checkpoints/best_model.pt \
  --video-dir data/split/test \
  --T-in 30 \
  --num-predict 60 \
  --video-idx 5
```

### Monitor Training
```bash
# View training curves and metrics
tensorboard --logdir runs
```

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

## 💡 Training Tips

### For Best Results:
1. **Start simple**: Train on custom CNN first, then try pretrained encoders
2. **Batch size**: 4-8 (depending on GPU memory)
3. **Learning rate**: 1e-4 with cosine annealing scheduler
4. **Gradient clipping**: 1.0 prevents exploding gradients
5. **Data augmentation**: Enabled by default for training set
6. **Validation**: Monitor every epoch, save best model automatically

### Model Selection:
- **GRU**: Fastest, ~17M params, good for quick experiments
- **Transformer**: Better long-term predictions, ~22M params
- **Pretrained encoder**: Best quality, requires GPU for perceptual loss

### GPU vs CPU:
- **CPU**: Use `--no-perceptual` flag, smaller batch size (2-4)
- **GPU**: Enable perceptual loss for better visual quality
- **MPS (Mac M1/M2)**: Use `--device mps`, similar to CPU settings

### Expected Training Time:
- **100 epochs on CPU**: ~8-12 hours (GRU model)
- **100 epochs on GPU**: ~2-4 hours (with perceptual loss)
- **Convergence**: Loss stabilizes around epoch 50-80

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

## 📊 Expected Results

### After 100 Epochs:
- **Short-term (1-10 frames)**: High accuracy, MSE < 100
- **Medium-term (10-20 frames)**: Good physics, slight blurring
- **Long-term (20-30+ frames)**: Reasonable trajectories, visible degradation

### What Good Predictions Look Like:
✅ **Physics consistency**: Balls bounce realistically, no teleporting
✅ **Smooth motion**: No sudden jumps between frames
✅ **Temporal coherence**: Low temporal consistency loss
✅ **Visual quality**: Clear shapes, minimal blurring (with pretrained encoder)

### Common Issues:
❌ **Blurry predictions**: Increase perceptual loss weight or use pretrained encoder
❌ **Mode collapse**: Predictions converge to mean frame - reduce learning rate
❌ **Physics violations**: Increase training epochs or use transformer model
❌ **Exploding gradients**: Enable gradient clipping (default: 1.0)

## ❓ FAQ

**Q: Can I use my own videos?**
A: Yes! Place `.mp4` files in `data/raw/` and run `split_dataset.py`. The model works best with physics-based or predictable motion.

**Q: How much GPU memory do I need?**
A: Minimum 4GB for batch_size=4. Recommended 8GB+ for batch_size=8 with pretrained encoders.

**Q: Why are my predictions blurry?**
A: Try: (1) Use pretrained encoder like DINOv2, (2) Enable perceptual loss, (3) Increase training epochs.

**Q: Which world model should I use?**
A: Start with GRU (fast), upgrade to Transformer for better long-term predictions.

**Q: Can I train on CPU?**
A: Yes, but slower. Use `--device cpu --no-perceptual --batch-size 2` for faster training.

**Q: How do I resume training?**
A: Use `--resume checkpoints/checkpoint_epoch_N.pt`

**Q: Where are the outputs saved?**
A: Checkpoints in `checkpoints/`, logs in `runs/`, demos in `outputs/demos/`

## 🙏 Acknowledgments

Inspired by NVIDIA Cosmos and world models research.

## 📄 License

MIT License - feel free to use for learning and portfolio projects!
