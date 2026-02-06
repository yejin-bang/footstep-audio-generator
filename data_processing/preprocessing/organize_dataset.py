import json
import shutil
from pathlib import Path

def organize_dataset(processed_clips_dir: str, metadata_dir: str, output_dir: str):
    """
    Copy clips to train/val/test folders based on metadata.
    
    Args:
        processed_clips_dir: Directory containing processed audio clips
        metadata_dir: Directory containing metadata JSONL files
        output_dir: Output directory for organized dataset
    """
    output_path = Path(output_dir)
    
    for split in ['train', 'val', 'test']:
        # Read metadata
        jsonl_path = Path(metadata_dir) / f"{split}.jsonl"
        
        if not jsonl_path.exists():
            print(f"⚠️  Warning: {jsonl_path} not found, skipping {split}")
            continue
            
        with open(jsonl_path, 'r') as f:
            clips = [json.loads(line) for line in f]
        
        # Create split directory
        split_dir = output_path / 'audio' / split
        split_dir.mkdir(parents=True, exist_ok=True)
        
        # Get unique filenames (3x captions per file)
        unique_files = set(clip['file_name'] for clip in clips)
        
        print(f"\nCopying {len(unique_files)} files to {split}/...")
        
        copied = 0
        not_found = []
        
        for filename in unique_files:
            # Find source file in processed clips
            source = Path(processed_clips_dir)
            # Search recursively
            source_files = list(source.rglob(filename))
            
            if source_files:
                src = source_files[0]
                dst = split_dir / filename
                shutil.copy2(src, dst)
                copied += 1
            else:
                not_found.append(filename)
        
        print(f"  ✅ Copied {copied} files")
        
        if not_found:
            print(f"  ⚠️  {len(not_found)} files not found:")
            for f in not_found[:5]:  # Show first 5
                print(f"     - {f}")
            if len(not_found) > 5:
                print(f"     ... and {len(not_found) - 5} more")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Organize dataset into train/val/test folders')
    parser.add_argument(
        '--processed_clips_dir',
        required=True,
        help='Directory with processed clips'
    )
    parser.add_argument(
        '--metadata_dir',
        required=True,
        help='Directory with metadata JSONL files'
    )
    parser.add_argument(
        '--output_dir',
        required=True,
        help='Output directory for organized dataset'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.processed_clips_dir).exists():
        print(f"❌ ERROR: Processed clips directory not found: {args.processed_clips_dir}")
        exit(1)
    
    if not Path(args.metadata_dir).exists():
        print(f"❌ ERROR: Metadata directory not found: {args.metadata_dir}")
        exit(1)
    
    organize_dataset(
        processed_clips_dir=args.processed_clips_dir,
        metadata_dir=args.metadata_dir,
        output_dir=args.output_dir,
        audio_subdir=args.audio_subdir
    )
    
    print("\n✅ Dataset organization complete!")