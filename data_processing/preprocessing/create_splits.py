import json
from pathlib import Path
from typing import Dict, List
import numpy as np

def load_preprocessing_metadata(metadata_path: str) -> Dict:
    """Load the audio_segments_metadata.json from preprocess_audio.py"""
    with open(metadata_path, 'r') as f:
        return json.load(f)

def generate_jsonl_metadata(preprocessing_metadata: Dict, output_dir: str, parent_recording_field: str = 'source_file'):
    """Generate train.jsonl, val.jsonl, test.jsonl with proper splits"""
    clips = preprocessing_metadata['all_clips_metadata']
    
    # Group clips by parent recording
    parent_groups = {}
    for clip in clips:
        parent = clip.get(parent_recording_field, clip['clip_filename'])
        if parent not in parent_groups:
            parent_groups[parent] = []
        parent_groups[parent].append(clip)
    
    # Split parent recordings 80/10/10
    parent_list = list(parent_groups.keys())
    np.random.seed(42)  
    np.random.shuffle(parent_list)
    
    n_total = len(parent_list)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    
    train_parents = parent_list[:n_train]
    val_parents = parent_list[n_train:n_train+n_val]
    test_parents = parent_list[n_train+n_val:]
    
    print(f"Split {n_total} parent recordings:")
    print(f"  Train: {len(train_parents)} parents")
    print(f"  Val: {len(val_parents)} parents")
    print(f"  Test: {len(test_parents)} parents")
    
    # Assign clips to splits
    train_clips = []
    val_clips = []
    test_clips = []
    
    for clip in clips:
        parent = clip.get(parent_recording_field, clip['clip_filename'])
        
        # Create base metadata entry
        entry = {
            'file_name': clip['clip_filename'],
            'caption': '',  # Will add captions later
            'duration': clip['duration'],
            'clip_index': clip['clip_index']
        }
        
        # Assign to split
        if parent in train_parents:
            entry['split'] = 'train'
            train_clips.append(entry)
        elif parent in val_parents:
            entry['split'] = 'val'
            val_clips.append(entry)
        else:
            entry['split'] = 'test'
            test_clips.append(entry)
    
    # Save JSONL files
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save train.jsonl
    with open(output_path / 'train.jsonl', 'w') as f:
        for clip in train_clips:
            f.write(json.dumps(clip) + '\n')
    
    # Save val.jsonl
    with open(output_path / 'val.jsonl', 'w') as f:
        for clip in val_clips:
            f.write(json.dumps(clip) + '\n')
    
    # Save test.jsonl  
    with open(output_path / 'test.jsonl', 'w') as f:
        for clip in test_clips:
            f.write(json.dumps(clip) + '\n')
    
    print(f"\nSaved metadata:")
    print(f"  {output_path / 'train.jsonl'} - {len(train_clips)} clips")
    print(f"  {output_path / 'val.jsonl'} - {len(val_clips)} clips")
    print(f"  {output_path / 'test.jsonl'} - {len(test_clips)} clips")
    
    return {
        'train': train_clips,
        'val': val_clips,
        'test': test_clips
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate training metadata splits')
    parser.add_argument('--metadata', required=True, help='Path to audio_segments_metadata.json')
    parser.add_argument('--output_dir', required=True, help='Output directory for JSONL files')
    
    args = parser.parse_args()
    
    preprocessing_metadata = load_preprocessing_metadata(args.metadata)
    splits = generate_jsonl_metadata(preprocessing_metadata, args.output_dir)
    
    print(f"\n✅ Metadata generation complete!")
    print(f"  Total clips: {len(splits['train']) + len(splits['val']) + len(splits['test'])}")