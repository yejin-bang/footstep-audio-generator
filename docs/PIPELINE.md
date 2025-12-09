# Complete Footstep Audio Generation Pipeline 🎬→🔊

## 📋 Overview

This is your **complete end-to-end pipeline** that takes a video file and outputs footstep audio.

**Input:** Video file (`.mp4`, `.avi`, `.mov`, etc.)  
**Output:** Audio file (`.wav`) with exact video duration

---

## 🎯 Pipeline Flow

```
Video File
    ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: Video Validation                                    │
│ ✓ Check file exists and is readable                         │
│ ✓ Extract: duration, FPS, resolution                        │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: Footstep Detection                                  │
│ ✓ MediaPipe pose estimation                                 │
│ ✓ Hip-heel distance calculation                             │
│ ✓ Peak detection + alternation filter                       │
│ Output: [(0.5s, LEFT), (1.2s, RIGHT), ...]                  │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: Scene Analysis                                      │
│ ✓ CLIP-based environment classification                     │
│ ✓ Determine: surface, footwear, sound characteristics       │
│ Output: "person walking on marble with dress shoes"         │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 4: Audio Generation (Pluggable Backends)               │
│ ✓ RunPod backend: serverless GPU (default)                  │
│ ✓ Mock backend: synthetic audio for testing                 │
│ ✓ Uses LoRA-tuned Stable Audio Open model                   │
│ ✓ Each variation: ~6 seconds of footstep audio              │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 5: Spatial Audio Processing                            │
│ ✓ Randomly assign variations to timestamps                  │
│ ✓ Apply stereo panning (L/R based on foot side)             │
│ ✓ Apply volume attenuation (distance-based)                 │
│ ✓ Mix all footsteps into single stereo track                │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 6: Finalization & Video Merging                         │
│ ✓ Match exact video duration                                │
│ ✓ Apply fade in/out                                         │
│ ✓ Normalize to -1dB peak                                    │
│ ✓ Export WAV file                                            │
│ ✓ Optional: Merge audio with video (--merge-video)          │
└─────────────────────────────────────────────────────────────┘
    ↓
Audio File (exact video duration)
```

---

## 🚀 Quick Start

### **Prerequisites**

1. **Install dependencies:**
```bash
# Install from requirements.txt
pip install -r requirements.txt

# Or install package in development mode
pip install -e .
```

2. **Setup RunPod (for production mode):**
```bash
# Create .env file
touch .env

# Add your RunPod credentials
echo "RUNPOD_API_KEY=your_api_key_here" >> .env
echo "RUNPOD_ENDPOINT_URL=https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run" >> .env
```

3. **Install ffmpeg (for video merging):**
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg
```

### **Basic Usage**

```bash
# Process a video (RunPod backend - default)
python -m src.main_pipeline video.mp4

# Output will be in: pipeline_outputs/video_footsteps.wav
```

### **Custom Output Path**

```bash
python -m src.main_pipeline video.mp4 --output my_footsteps.wav
```

### **Mock Backend (No API Required)**

```bash
python -m src.main_pipeline video.mp4 --backend mock
```

### **Video Merging**

```bash
python -m src.main_pipeline video.mp4 --merge-video
```

---

## ⚙️ Configuration Options

### **Command Line Arguments**

| Argument | Default | Description |
|----------|---------|-------------|
| `video_path` | *required* | Path to input video file |
| `--output` | *auto* | Output audio file path |
| `--backend` | `runpod` | Backend: `runpod` (default) or `mock` (testing) |
| `--variations` | `1` | Number of audio variations |
| `--audio-length` | `6.0` | Length of each variation (seconds) |
| `--cfg-scale` | `7.0` | CFG scale for generation |
| `--steps` | `100` | Diffusion steps |
| `--output-dir` | `./pipeline_outputs` | Output directory |
| `--no-intermediates` | *off* | Don't save intermediate files |
| `--merge-video` | *off* | Merge audio with video (requires ffmpeg) |
| `--no-merge-video` | *off* | Don't merge video (audio only) |

### **Example Commands**

```bash
# Fast mode (fewer steps, good for testing)
python -m src.main_pipeline video.mp4 --backend mock --steps 50 --variations 5

# High quality mode
python -m src.main_pipeline video.mp4 --backend runpod --steps 150 --cfg-scale 8.0 --variations 10

# Custom output with video merging
python -m src.main_pipeline video.mp4 \
  --output final_audio.wav \
  --backend runpod \
  --variations 8 \
  --steps 100 \
  --merge-video
```

---

## 📂 Output Structure

```
pipeline_outputs/
├── video_footsteps.wav              # Final audio output
├── video_footsteps.json             # Metadata (if save_intermediates=True)
└── variations/                      # Generated audio variations
    ├── 20241113_footstep_1.wav
    ├── 20241113_footstep_2.wav
    └── ...
```

### **Metadata JSON Example**

```json
{
  "success": true,
  "video_path": "video.mp4",
  "output_audio_path": "pipeline_outputs/video_footsteps.wav",
  "processing_time_seconds": 245.3,
  "video_info": {
    "duration": 10.5,
    "fps": 30.0,
    "width": 1920,
    "height": 1080
  },
  "num_footsteps": 18,
  "num_audio_variations": 8,
  "audio_prompt": "person walking on marble floor with dress shoes, high-quality"
}
```

---

## 🔧 Python API Usage

### **Basic Example**

```python
from main_pipeline import FootstepAudioPipeline, PipelineConfig

# Create pipeline with default settings
pipeline = FootstepAudioPipeline()

# Process video
results = pipeline.process_video("video.mp4")

# Check results
if results['success']:
    print(f"✓ Audio saved: {results['output_audio_path']}")
    print(f"✓ Processing time: {results['processing_time_seconds']:.1f}s")
    print(f"✓ Footsteps detected: {results['num_footsteps']}")
```

### **Custom Configuration**

```python
from main_pipeline import FootstepAudioPipeline, PipelineConfig

# Create custom configuration
config = PipelineConfig(
    use_gpu="runpod",           # or "local"
    audio_variations=10,        # More variations
    audio_length=6.0,
    cfg_scale=8.0,              # Higher quality
    steps=150,                  # More steps
    output_dir="./my_outputs",
    save_intermediates=True,
    create_visualizations=True
)

# Create pipeline
pipeline = FootstepAudioPipeline(config)

# Process video
results = pipeline.process_video(
    video_path="video.mp4",
    output_audio_path="custom_output.wav"
)
```

### **Batch Processing**

```python
from main_pipeline import FootstepAudioPipeline
from pathlib import Path

# Create pipeline once
pipeline = FootstepAudioPipeline()

# Process multiple videos
video_dir = Path("videos")
for video_path in video_dir.glob("*.mp4"):
    print(f"Processing: {video_path.name}")
    
    try:
        results = pipeline.process_video(str(video_path))
        print(f"  ✓ Success: {results['output_audio_path']}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
```

---

## 🎛️ Component Details

### **1. Video Validator**
- **Input:** Video file path
- **Output:** Video metadata (duration, FPS, resolution)
- **Validates:** File exists, readable, has valid frames

### **2. Footstep Detector**
- **Input:** Video file path
- **Output:** List of `(timestamp, foot_side)` pairs
- **Method:** MediaPipe pose → hip-heel distance → peak detection → alternation filter
- **Performance:** ~64.9% F1-score

### **3. Scene Analyzer**
- **Input:** Video frames
- **Output:** Audio prompt (e.g., "boots on concrete")
- **Method:** CLIP-based environment classification
- **Supports:** Multiple scene segments (detects transitions)

### **4. Audio Generator**
- **Input:** Text prompt
- **Output:** Audio tensor + saved WAV file
- **Modes:** 
  - **RunPod:** Serverless GPU (recommended for production)
  - **Local:** Your machine's GPU/CPU
- **Model:** LoRA-tuned Stable Audio Open 1.0

### **5. Spatial Audio Processor**
- **Input:** Audio variations + spatial data
- **Output:** Final spatialized stereo audio
- **Processing:**
  - Random assignment of variations to timestamps
  - Constant power panning (L/R)
  - Volume attenuation (distance-based)
  - Final mixing + normalization

---

## 🐛 Troubleshooting

### **Error: "No footsteps detected"**

**Causes:**
- Person not visible in video
- Low video quality / resolution
- Person not walking (standing still)

**Solutions:**
- Check video has person walking
- Ensure person is fully visible
- Try different confidence threshold: `--confidence 0.5`

---

### **Error: "RunPod mode requested but runpod_api not available"**

**Solution:**
```bash
# Make sure these files exist:
ls runpod_api.py  # Should exist
ls .env           # Should exist

# Install dependencies
pip install python-dotenv requests

# Check .env has API key
cat .env
```

---

### **Error: "HTTP 401: Unauthorized"**

**Solution:**
```bash
# Your RunPod API key is wrong or expired
# Get new key from: https://www.runpod.io/console/user/settings

# Edit .env and update key
nano .env
```

---

### **Slow Processing**

**Tips:**
- Use `--steps 50` for faster generation (lower quality)
- Use `--variations 5` to generate fewer variations
- Use `--use-gpu local` if you have a fast GPU
- Process shorter videos first to test

**Expected times:**
- 10-second video: ~3-5 minutes (RunPod, 8 variations, 100 steps)
- 30-second video: ~5-8 minutes
- 60-second video: ~8-12 minutes

---

### **Audio Quality Issues**

**If audio sounds robotic/artificial:**
- Increase CFG scale: `--cfg-scale 8.0`
- Increase steps: `--steps 150`
- Generate more variations: `--variations 10`

**If footsteps are too loud/quiet:**
- Edit `PipelineConfig.min_attenuation_db` and `max_attenuation_db`
- Or manually adjust output audio in DAW

---

## 💡 Tips & Best Practices

### **Video Requirements**

✅ **Good:**
- Person clearly visible
- Walking continuously
- Good lighting
- 720p+ resolution
- 24-30 FPS

❌ **Avoid:**
- Extreme close-ups (can't see feet)
- Low resolution (<480p)
- Person standing still
- Multiple people (detector may get confused)

### **Performance Optimization**

**For Testing:**
```bash
python main_pipeline.py video.mp4 \
  --steps 50 \
  --variations 5 \
  --audio-length 3.0
```

**For Production:**
```bash
python main_pipeline.py video.mp4 \
  --steps 150 \
  --cfg-scale 8.0 \
  --variations 10
```

### **Cost Optimization (RunPod)**

- **Fewer variations** = Lower cost
- **Fewer steps** = Faster + cheaper
- **Shorter audio length** = Lower cost

Example costs (RTX A5000):
- 8 variations × 100 steps × 6s = ~$0.05
- 10 variations × 150 steps × 6s = ~$0.10
- 5 variations × 50 steps × 3s = ~$0.02

---

## 📊 Example Output

### **Console Output**

```
================================================================================
FOOTSTEP AUDIO GENERATION PIPELINE
================================================================================
Mode: RUNPOD
Audio variations: 8
Output directory: pipeline_outputs

Initializing pipeline components...
✓ All components initialized

================================================================================
PROCESSING: walk_video.mp4
================================================================================

STEP 1: Video Validation
--------------------------------------------------------------------------------
============================================================
Video Validation Starting...
File: walk_video.mp4
Size: 15.3 MB
Duration: 10.5s (315 frames)
Resolution: 1920x1080
FPS: 30.0
Video validation successful!
============================================================

STEP 2: Footstep Detection
--------------------------------------------------------------------------------
============================================================
SIMPLE FOOTSTEP DETECTION
============================================================
Architecture: Video -> Pose -> Distances -> Peaks -> Filter -> Timestamps

[Detection process output...]

✓ Detected 18 footsteps

STEP 3: Scene Analysis
--------------------------------------------------------------------------------
Starting multi-segment scene analysis...
Detected 1 scene segments
Analyzing segment 0: 0.0s - 10.5s (10.5s)
  Environment: indoor hallway (0.847)
  Prompt: 'footstep sound effect, leather dress shoes on marble, high-quality'

✓ Generated prompt: 'footstep sound effect, leather dress shoes on marble, high-quality'

STEP 4: Audio Generation
--------------------------------------------------------------------------------
Generating 8 variations...
Prompt: 'footstep sound effect, leather dress shoes on marble, high-quality'
Mode: runpod

[Generation progress for each variation...]

✓ Generated 8 audio variations

STEP 5: Preparing Spatial Audio Data
--------------------------------------------------------------------------------
✓ Prepared spatial data for 18 footsteps

STEP 6: Spatial Audio Processing
--------------------------------------------------------------------------------
Creating final mix...
  Video duration: 10.50s
  Sample rate: 44100Hz
  Footsteps to place: 18

  ✓ Mixed 18 footsteps
  ✓ Peak level: -1.0dB

✓ Final audio saved: pipeline_outputs/walk_video_footsteps.wav

================================================================================
PIPELINE COMPLETE
================================================================================
Total processing time: 245.32s
Video duration: 10.50s
Footsteps detected: 18
Audio variations generated: 8
Output audio: pipeline_outputs/walk_video_footsteps.wav
```

---

## 🎓 Understanding the Pipeline

### **Why 8 variations?**

Real footsteps have natural variation - no two steps sound exactly the same. By generating multiple variations and randomly assigning them, the output sounds more natural and less repetitive.

### **Why RunPod by default?**

- **Scalable:** Can process multiple videos in parallel
- **Fast:** RTX A5000 GPU is much faster than most local machines
- **Cost-effective:** Only pay for GPU time used (~$0.006 per 6s audio)
- **No setup:** No need to install large models locally

### **What does spatial audio do?**

- **Panning:** Places footsteps in stereo field (left foot → left channel, right foot → right channel)
- **Attenuation:** Adjusts volume based on distance from camera
- **Mixing:** Combines all footsteps into coherent audio track

---

## 📚 Related Documentation

- **Audio Generator Guide:** `Audio_Generator_Modification_Guide.md`
- **RunPod Setup:** `RUNPOD_SETUP_GUIDE.md`
- **Architecture Overview:** `HOW_FILES_WORK_TOGETHER.md`
- **Quick Start:** `00_START_HERE.md`

---

## ✅ Checklist Before Running

- [ ] All Python files in same directory
- [ ] Dependencies installed: `pip install -r requirements.txt`
- [ ] `.env` file created with RunPod API key (if using RunPod mode)
- [ ] Test video file available
- [ ] Enough disk space for output files (~50MB per 10s video)

---

## 🎉 You're Ready!

**Test the pipeline:**

```bash
python main_pipeline.py your_video.mp4
```

**Expected result:**
- Audio file created in `pipeline_outputs/`
- Same duration as your video
- Natural-sounding footsteps
- Proper stereo positioning

**Questions? Need help?**
- Check the troubleshooting section above
- Review the component documentation
- Ask for help!

---

## 🚀 Next Steps

1. **Test on short video** (5-10 seconds) first
2. **Review output audio** - does it sound good?
3. **Adjust parameters** if needed (steps, cfg_scale, variations)
4. **Process longer videos** once satisfied
5. **Integrate into your workflow!**

**Happy footstep generating! 🎬→🔊**
