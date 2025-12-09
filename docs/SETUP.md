# 🔧 Quick Setup Guide

Complete setup guide for the Footstep Audio Generation Pipeline.

---

## 📋 Prerequisites

1. **Python 3.8+** installed
2. **Git** (for cloning CLIP)
3. **ffmpeg** (optional, for video merging)

---

## 🚀 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/sfx-project.git
cd sfx-project
```

### Step 2: Install Dependencies

```bash
# Install from requirements.txt
pip install -r requirements.txt

# Or install package in development mode
pip install -e .
```

**What gets installed:**
- **Core**: numpy, scipy, opencv-python, mediapipe
- **Audio**: soundfile, librosa
- **ML/AI**: torch, torchvision, CLIP
- **API**: requests, python-dotenv
- **Visualization**: matplotlib
- **Testing**: pytest, pytest-cov

### Step 3: Install ffmpeg (Optional, for Video Merging)

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

---

## ⚙️ Configuration

### Option 1: RunPod Backend (Recommended for Production)

Create a `.env` file in the project root:

```bash
# Create .env file
touch .env

# Add your RunPod credentials
echo "RUNPOD_API_KEY=your_api_key_here" >> .env
echo "RUNPOD_ENDPOINT_URL=https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run" >> .env
echo "RUNPOD_TIMEOUT=300" >> .env
```

**Where to get RunPod credentials:**
1. Sign up at https://runpod.io
2. Create a serverless endpoint with the audio generation model
3. Copy the API key and endpoint URL

### Option 2: Mock Backend (For Testing)

No configuration needed! The mock backend generates synthetic audio without any API.

---

## ✅ Verify Installation

### Test 1: Check Configuration

```bash
python -m src.config
```

**Expected output:**
```
================================================================================
FOOTSTEP AUDIO PIPELINE - CONFIGURATION
================================================================================

Project Structure:
  PROJECT_ROOT: /path/to/sfx-project
  SRC_DIR: /path/to/sfx-project/src
  CONFIG_DIR: /path/to/sfx-project/config
  ...

Validation:
  ✓ Configuration is valid
```

### Test 2: Run with Mock Backend

```bash
# Create a test video or use an existing one
python -m src.main_pipeline test_videos/walk1.mp4 --backend mock
```

**Expected output:**
```
🎵 Generating audio using 'mock' backend...
✅ Audio saved: generated_outputs/...
✓ Successfully merged video and audio!
```

### Test 3: Run Individual Components

```bash
# Test footstep detector
python -m src.footstep_detector

# Test scene analyzer
python -m src.scene_analyzer

# Test spatial audio processor
python -m src.spatial_audio_processor
```

---

## 📁 Project Structure

After setup, your project should look like:

```
sfx-project/
├── .env                      # RunPod credentials (you create this)
├── requirements.txt          # Python dependencies
├── setup.py                  # Package installer
├── README.md                 # Main documentation
├── CLAUDE.md                 # Technical reference
│
├── src/                      # Main pipeline code
│   ├── __init__.py
│   ├── main_pipeline.py      # End-to-end orchestration
│   ├── video_validator.py    # Video validation
│   ├── footstep_detector.py  # Footstep detection
│   ├── scene_analyzer.py     # CLIP scene analysis
│   ├── audio_generator.py    # Audio generation wrapper
│   ├── spatial_audio_processor.py  # Spatial audio processing
│   ├── video_merger.py       # Video-audio merging
│   ├── config.py             # Configuration management
│   ├── logger.py             # Logging system
│   └── audio_backends/       # Pluggable backends
│       ├── base.py           # Abstract base class
│       ├── runpod_backend.py # RunPod implementation
│       └── mock_backend.py   # Testing backend
│
├── utils/                    # Utility modules
│   ├── pose_extractor.py     # MediaPipe wrapper
│   ├── runpod_api.py         # RunPod API client
│   └── runpod_client.py      # Command-line client
│
├── config/                   # Configuration files
│   ├── caption_config.json   # Vocabulary for prompts
│   ├── model_config_lora.json  # LoRA configuration
│   └── scene_config.json     # Scene analysis config
│
├── models/                   # Model checkpoints
│   └── best.ckpt             # LoRA checkpoint (if using RunPod)
│
├── tests/                    # Unit tests
│   ├── test_backends.py
│   ├── test_config.py
│   └── test_logger.py
│
└── Output Directories (auto-created):
    ├── pipeline_outputs/     # Final results
    ├── generated_outputs/    # Raw audio generation
    └── logs/                 # Log files
```

---

## 🎯 Basic Usage

### Example 1: Quick Test (Mock Backend)

```bash
python -m src.main_pipeline video.mp4 --backend mock
```

### Example 2: Production (RunPod Backend)

```bash
python -m src.main_pipeline video.mp4 --backend runpod
```

### Example 3: With Video Merging

```bash
python -m src.main_pipeline video.mp4 --backend runpod --merge-video
```

### Example 4: Full Customization

```bash
python -m src.main_pipeline video.mp4 \
  --backend runpod \
  --output my_footsteps.wav \
  --variations 10 \
  --cfg-scale 8.0 \
  --steps 150 \
  --merge-video
```

---

## 🧪 Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=src --cov-report=html

# Skip tests that require API
pytest -m "not requires_api"
```

---

## 🔍 Troubleshooting

### Issue 1: "ModuleNotFoundError: No module named 'cv2'"

**Fix:** Install opencv-python
```bash
pip install opencv-python
```

### Issue 2: "ModuleNotFoundError: No module named 'src'"

**Fix:** Run as module, not as script
```bash
# ✗ Wrong
python src/main_pipeline.py video.mp4

# ✓ Correct
python -m src.main_pipeline video.mp4
```

### Issue 3: "RUNPOD_API_KEY not set"

**Fix:** Either:
1. Add to `.env` file (for RunPod backend)
2. Use mock backend instead: `--backend mock`

### Issue 4: "ffmpeg is not installed"

**Fix:** Install ffmpeg or don't use `--merge-video` flag
```bash
brew install ffmpeg  # macOS
sudo apt-get install ffmpeg  # Linux
```

### Issue 5: "FileNotFoundError: Config not found"

**Fix:** Make sure you're in the project root directory
```bash
pwd  # Should show /path/to/sfx-project
ls config/  # Should list caption_config.json, etc.
```

---

## 🎓 Next Steps

### For Development:

1. **Read the docs:**
   - `README.md` - Project overview
   - `CLAUDE.md` - Technical details
   - `tests/README.md` - Testing guide

2. **Explore components:**
   ```bash
   python -m src.footstep_detector  # Run detector
   python -m src.scene_analyzer     # Run scene analysis
   python -m src.config             # Check configuration
   ```

3. **Write tests:**
   ```bash
   pytest tests/test_backends.py -v
   ```

### For Production:

1. **Set up RunPod:**
   - Sign up at https://runpod.io
   - Deploy the audio generation model
   - Add credentials to `.env`

2. **Process videos:**
   ```bash
   python -m src.main_pipeline video.mp4 --backend runpod --merge-video
   ```

3. **Monitor logs:**
   ```bash
   tail -f logs/pipeline_*.log
   ```

---

## 💡 Tips

1. **Use mock backend for testing** - No API required, fast iteration
2. **Check configuration first** - Run `python -m src.config` to validate setup
3. **Enable logging** - Add `--verbose` flag for detailed output (if implemented)
4. **Test individual components** - Easier to debug than full pipeline
5. **Use virtual environments** - Isolate dependencies

---

## 📚 Additional Resources

- **Main README**: [README.md](../README.md)
- **Technical Docs**: [CLAUDE.md](../CLAUDE.md)
- **Testing Guide**: [tests/README.md](../tests/README.md)
- **API Docs**: Coming soon

---

## 🎉 You're Ready!

You now have a fully working footstep audio generation pipeline!

**Quick start:**
```bash
python -m src.main_pipeline test_videos/walk1.mp4 --backend mock
```

**Questions?** Check [CLAUDE.md](../CLAUDE.md) for detailed documentation.

**Happy generating!** 🚀
