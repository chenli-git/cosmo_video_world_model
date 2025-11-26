"""
Dataset for loading video clips with context (T_in) and target (T_out) frames.
"""
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Optional, List
import random


class VideoClipDataset(Dataset):
    """
    Dataset that loads videos and splits them into clips of T_in + T_out frames.
    Each sample contains:
    - context: T_in frames (input)
    - target: T_out frames (to predict)
    """
    
    def __init__(self,
                 video_dir: str,
                 T_in: int = 30,
                 T_out: int = 30,
                 stride: int = 5,
                 video_extensions: List[str] = ['.mp4', '.avi', '.mov'],
                 resize: Optional[Tuple[int, int]] = None,
                 augment: bool = False):
        """
        Args:
            video_dir: Directory containing video files
            T_in: Number of input/context frames
            T_out: Number of output/target frames to predict
            stride: Stride between clips when extracting from videos
            video_extensions: Valid video file extensions
            resize: Optional (width, height) to resize frames
            augment: Whether to apply data augmentation
        """
        self.video_dir = Path(video_dir)
        self.T_in = T_in
        self.T_out = T_out
        self.T_total = T_in + T_out
        self.stride = stride
        self.resize = resize
        self.augment = augment
        
        # Find all video files recursively
        self.video_files = []
        for ext in video_extensions:
            self.video_files.extend(self.video_dir.glob(f"**/*{ext}"))
        
        print(f"Found {len(self.video_files)} videos in {video_dir}")
        
        # Build index of all valid clips
        self.clips = []
        self._build_clip_index()
        
        print(f"Total clips: {len(self.clips)}")
    
    def _build_clip_index(self):
        """Build an index of all valid clips from all videos."""
        for video_path in self.video_files:
            try:
                cap = cv2.VideoCapture(str(video_path))
                num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                
                # Calculate how many clips can be extracted
                if num_frames >= self.T_total:
                    num_clips = (num_frames - self.T_total) // self.stride + 1
                    
                    for clip_idx in range(num_clips):
                        start_frame = clip_idx * self.stride
                        self.clips.append((video_path, start_frame))
                
            except Exception as e:
                print(f"Warning: Could not process {video_path}: {e}")
    
    def __len__(self) -> int:
        return len(self.clips)
    
    def _load_frames(self, video_path: Path, start_frame: int, num_frames: int) -> np.ndarray:
        """Load a sequence of frames from a video."""
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frames = []
        for _ in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize if specified
            if self.resize is not None:
                frame = cv2.resize(frame, self.resize)
            
            frames.append(frame)
        
        cap.release()
        
        # Pad if we didn't get enough frames
        while len(frames) < num_frames:
            frames.append(frames[-1])
        
        return np.array(frames, dtype=np.uint8)
    
    def _augment_frames(self, frames: np.ndarray) -> np.ndarray:
        """Apply data augmentation to frames."""
        # Random horizontal flip
        if random.random() > 0.5:
            frames = np.flip(frames, axis=2).copy()
        
        # Random brightness adjustment
        if random.random() > 0.5:
            brightness_factor = random.uniform(0.8, 1.2)
            frames = np.clip(frames * brightness_factor, 0, 255).astype(np.uint8)
        
        return frames
    
    def _normalize_frames(self, frames: np.ndarray) -> torch.Tensor:
        """
        Normalize frames from [0, 255] to [-1, 1] and convert to tensor.
        
        Args:
            frames: np.ndarray of shape (T, H, W, C) with values in [0, 255]
            
        Returns:
            tensor: torch.Tensor of shape (T, C, H, W) with values in [-1, 1]
        """
        # Convert to float and normalize to [0, 1]
        frames = frames.astype(np.float32) / 255.0
        
        # Normalize to [-1, 1]
        frames = frames * 2.0 - 1.0
        
        # Convert to tensor and reorder dimensions: (T, H, W, C) -> (T, C, H, W)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)
        
        return frames
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a clip with context and target frames.
        
        Returns:
            context: Tensor of shape (T_in, C, H, W) in [-1, 1]
            target: Tensor of shape (T_out, C, H, W) in [-1, 1]
        """
        video_path, start_frame = self.clips[idx]
        
        # Load all frames for this clip
        frames = self._load_frames(video_path, start_frame, self.T_total)
        
        # Apply augmentation if enabled
        if self.augment:
            frames = self._augment_frames(frames)
        
        # Split into context and target
        context_frames = frames[:self.T_in]
        target_frames = frames[self.T_in:]
        
        # Normalize and convert to tensors
        context = self._normalize_frames(context_frames)
        target = self._normalize_frames(target_frames)
        
        return context, target


def create_data_loader(
    video_dir: str,
    T_in: int = 30,
    T_out: int = 30,
    batch_size: int = 4,
    stride: int = 5,
    resize: Optional[Tuple[int, int]] = (64, 64),
    num_workers: int = 4,
    shuffle: bool = True,
    augment: bool = False,
    drop_last: bool = True
) -> DataLoader:
    """
    Create a data loader for a single video directory (train, val, or test).
    
    Args:
        video_dir: Directory containing video files (e.g., 'data/split/train')
        T_in: Number of input frames
        T_out: Number of output frames
        batch_size: Batch size
        stride: Stride between clips
        resize: Target resolution (width, height)
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle the data
        augment: Whether to apply data augmentation
        drop_last: Whether to drop the last incomplete batch
        
    Returns:
        DataLoader for the specified directory
    """
    # Create dataset
    dataset = VideoClipDataset(
        video_dir=video_dir,
        T_in=T_in,
        T_out=T_out,
        stride=stride,
        resize=resize,
        augment=augment
    )
    
    # Create data loader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last
    )
    
    return loader


def denormalize_frames(frames: torch.Tensor) -> np.ndarray:
    """
    Convert normalized tensor frames back to uint8 numpy arrays.
    
    Args:
        frames: Tensor of shape (B, T, C, H, W) or (T, C, H, W) in [-1, 1]
        
    Returns:
        frames: np.ndarray of shape (..., T, H, W, C) in [0, 255]
    """
    # Move to CPU if needed
    if frames.is_cuda:
        frames = frames.cpu()
    
    # Denormalize from [-1, 1] to [0, 1]
    frames = (frames + 1.0) / 2.0
    
    # Clip and scale to [0, 255]
    frames = torch.clamp(frames, 0, 1)
    frames = (frames * 255).byte()
    
    # Convert to numpy and reorder dimensions
    frames = frames.numpy()
    
    # Handle different input shapes
    if frames.ndim == 5:  # (B, T, C, H, W)
        frames = frames.transpose(0, 1, 3, 4, 2)  # -> (B, T, H, W, C)
    elif frames.ndim == 4:  # (T, C, H, W)
        frames = frames.transpose(0, 2, 3, 1)  # -> (T, H, W, C)
    
    return frames


if __name__ == "__main__":
    # Test the dataset with train/val/test split
    print("Testing VideoClipDataset with separate folders...")
    
    import os
    
    # Test train loader
    train_dir = "data/split/train"
    if os.path.exists(train_dir):
        print(f"\nCreating train loader from {train_dir}...")
        train_loader = create_data_loader(
            video_dir=train_dir,
            T_in=30,
            T_out=30,
            batch_size=2,
            stride=5,
            resize=(64, 64),
            num_workers=0,
            shuffle=True,
            augment=True  # Enable augmentation for training
        )
        
        context_batch, target_batch = next(iter(train_loader))
        print(f"Train batch shapes:")
        print(f"  Context: {context_batch.shape}")  # (B, T_in, C, H, W)
        print(f"  Target: {target_batch.shape}")    # (B, T_out, C, H, W)
    
    # Test val loader
    val_dir = "data/split/val"
    if os.path.exists(val_dir):
        print(f"\nCreating val loader from {val_dir}...")
        val_loader = create_data_loader(
            video_dir=val_dir,
            T_in=30,
            T_out=30,
            batch_size=2,
            stride=5,
            resize=(64, 64),
            num_workers=0,
            shuffle=False,
            augment=False
        )
        
        context_batch, target_batch = next(iter(val_loader))
        print(f"Val batch shapes:")
        print(f"  Context: {context_batch.shape}")
        print(f"  Target: {target_batch.shape}")
    
    # Test test loader
    test_dir = "data/split/test"
    if os.path.exists(test_dir):
        print(f"\nCreating test loader from {test_dir}...")
        test_loader = create_data_loader(
            video_dir=test_dir,
            T_in=30,
            T_out=30,
            batch_size=2,
            stride=5,
            resize=(64, 64),
            num_workers=0,
            shuffle=False,
            augment=False
        )
        
        context_batch, target_batch = next(iter(test_loader))
        print(f"Test batch shapes:")
        print(f"  Context: {context_batch.shape}")
        print(f"  Target: {target_batch.shape}")
    
    print("\n✅ All tests passed!")
