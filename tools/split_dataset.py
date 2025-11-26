"""
Split videos into train/val/test folders.
"""
import shutil
import argparse
from pathlib import Path
import random
from tqdm import tqdm


def split_videos(input_dir: str,
                output_dir: str,
                train_ratio: float = 0.8,
                val_ratio: float = 0.1,
                test_ratio: float = 0.1,
                seed: int = 42):
    """
    Split videos from input_dir into train/val/test subdirectories.
    
    Args:
        input_dir: Directory containing all videos
        output_dir: Output directory to create train/val/test folders
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Find all video files
    video_extensions = ['.mp4', '.avi', '.mov']
    video_files = []
    for ext in video_extensions:
        video_files.extend(input_path.glob(f"**/*{ext}"))
    
    print(f"Found {len(video_files)} videos in {input_dir}")
    
    if len(video_files) == 0:
        print("No videos found!")
        return
    
    # Shuffle videos
    random.shuffle(video_files)
    
    # Calculate split sizes
    total = len(video_files)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    
    train_files = video_files[:train_size]
    val_files = video_files[train_size:train_size + val_size]
    test_files = video_files[train_size + val_size:]
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_files)} videos ({len(train_files)/total*100:.1f}%)")
    print(f"  Val:   {len(val_files)} videos ({len(val_files)/total*100:.1f}%)")
    print(f"  Test:  {len(test_files)} videos ({len(test_files)/total*100:.1f}%)")
    
    # Create output directories
    train_dir = output_path / "train"
    val_dir = output_path / "val"
    test_dir = output_path / "test"
    
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy files to respective directories
    print("\nCopying files...")
    
    print("Copying train videos...")
    for video_file in tqdm(train_files):
        # Preserve subdirectory structure (e.g., bouncing/, pendulum/)
        relative_path = video_file.relative_to(input_path)
        dest = train_dir / relative_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(video_file, dest)
    
    print("Copying validation videos...")
    for video_file in tqdm(val_files):
        relative_path = video_file.relative_to(input_path)
        dest = val_dir / relative_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(video_file, dest)
    
    print("Copying test videos...")
    for video_file in tqdm(test_files):
        relative_path = video_file.relative_to(input_path)
        dest = test_dir / relative_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(video_file, dest)
    
    print(f"\n✅ Dataset split complete!")
    print(f"📁 Output directory: {output_dir}")
    print(f"   - train/: {len(train_files)} videos")
    print(f"   - val/:   {len(val_files)} videos")
    print(f"   - test/:  {len(test_files)} videos")


def main():
    parser = argparse.ArgumentParser(
        description="Split videos into train/val/test folders"
    )
    parser.add_argument("--input", type=str, default="data/raw",
                       help="Input directory containing videos")
    parser.add_argument("--output", type=str, default="data/split",
                       help="Output directory for train/val/test folders")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                       help="Proportion for training (default: 0.8)")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                       help="Proportion for validation (default: 0.1)")
    parser.add_argument("--test-ratio", type=float, default=0.1,
                       help="Proportion for testing (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        print(f"Warning: Ratios sum to {total_ratio}, not 1.0")
        print("Normalizing ratios...")
        args.train_ratio /= total_ratio
        args.val_ratio /= total_ratio
        args.test_ratio /= total_ratio
    
    split_videos(
        args.input,
        args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
