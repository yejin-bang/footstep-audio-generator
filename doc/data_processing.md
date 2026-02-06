# Data Processing Pipeline

Audio preprocessing and dataset preparation scripts for footstep audio training.

## Scripts Overview

### 1. scan_audio_assets.py
Scans raw audio assets and creates inventory.
```bash
python scan_audio_assets.py --assets_path data/raw_audio --output data/asset_inventory.json --verbose
```

### 2. preprocess_audio.py
Processes audio files with peak detection and silence trimming.
```bash
python preprocess_audio.py --inventory data/asset_inventory.json --output data/preprocessed --visualize 3
```

### 3. create_splits.py
Creates train/val/test splits (80%/10%/10%).
```bash
python create_splits.py --metadata data/preprocessed/peak_width_metadata.json --output_dir data/metadata
```

### 4. generate_captions.py
Generates three captions for each audio clip.
```bash
python generate_captions.py --config config/caption_config.json --metadata_dir data/metadata
```

### 5. organize_dataset.py
Organizes final dataset into train/val/test folders.
```bash
python organize_dataset.py --processed_clips_dir data/preprocessed --metadata_dir data/metadata --output_dir data/final_dataset
```

## Full Pipeline Example
```bash
# Step 1: Scan assets
python scan_audio_assets.py --assets_path data/raw_audio

# Step 2: Preprocess audio
python preprocess_audio.py --inventory data/asset_inventory.json --output data/preprocessed

# Step 3: Create splits
python create_splits.py --metadata data/preprocessed/peak_width_metadata.json --output_dir data/metadata

# Step 4: Generate captions
python generate_captions.py --config config/caption_config.json --metadata_dir data/metadata

# Step 5: Organize dataset
python organize_dataset.py --processed_clips_dir data/preprocessed --metadata_dir data/metadata --output_dir data/final_dataset
```