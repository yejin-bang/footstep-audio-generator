# AI Footstep Generator

**Automatically generate realistic, spatialized footstep audio from a video using AI.**

[![Demo Video](https://img.youtube.com/vi/2v67SE0sxHg/maxresdefault.jpg)](https://youtu.be/2v67SE0sxHg)

**[▶️ Watch 40-second Demo](https://youtu.be/2v67SE0sxHg)**

---

## What It Does

Upload a video of someone walking → Get back the same video with realistic footstep sounds automatically synced and spatialized (left/right panning, distance attenuation).

The system:
1. **Detects footsteps** using pose estimation and signal processing
2. **Understands the scene** (indoor/outdoor, surface type) using computer vision
3. **Generates contextual audio** using a fine-tuned AI audio model
4. **Applies spatial processing** for realistic left/right panning and depth

---

## Quick Start

```bash
# Install dependencies
git clone https://github.com/yejin-bang/ai-footstep-generator.git
cd ai-footstep-generator
pip install -r requirements.txt

# Test with mock backend (no API key needed)
python -m src.main_pipeline your_video.mp4 --backend mock --merge-video

# Production mode (requires RunPod API key)
cp .env.example .env
# Add your RunPod API key to .env
# Get your key at: https://www.runpod.io/console/user/settings
python -m src.main_pipeline your_video.mp4 --merge-video
```

**Output:** `outputs/your_video_with_footsteps.mp4`

---

## Results

### Audio Quality (Blind A/B Testing)

| Metric | Score | Details |
|--------|-------|---------|
| **Perceptual Quality** | **3.54/5.0** | Blind test vs. base model (91 samples) |
| Best Performance | 4.25/5.0 | Dress shoes on wood, heels on metal |
| Training Data | 4,023 samples | LoRA fine-tuning (2.3% of base model params) |

### Detection Performance

| Component | Metric | Value |
|-----------|--------|-------|
| **Footstep Detection** | F1-Score | 65% (avg) |
| Performance Range | 45-85% | Varies by video quality/angle |
| Precision | ~70% | Low false positive rate |
| Recall | ~60% | Catches most visible steps |
| **Scene Classification** | Accuracy | ~95% |
| Processing Speed | ~1 min | Per 10-second video (A5000 GPU) |

---

## Technical Architecture

**Pipeline Stages:**
1. **Video Validation** (OpenCV) → Extract metadata
2. **Footstep Detection** (MediaPipe) → Hip-heel distance analysis + peak detection
3. **Scene Analysis** (CLIP) → Environment classification (15 categories)
4. **Audio Generation** (LoRA-tuned Stable Audio Open) → Context-aware footstep sounds
5. **Spatial Processing** (NumPy/Librosa) → Constant-power panning + distance attenuation
6. **Video Merging** (FFmpeg) → Combine audio and video

**Tech Stack:**
- **Computer Vision:** MediaPipe (pose), CLIP (scene understanding)
- **Audio Generation:** Stable Audio Open + LoRA fine-tuning
- **Signal Processing:** NumPy, Librosa, SciPy
- **Infrastructure:** Serverless GPU (RunPod), PyTorch
- **Web Interface:**
  - Backend: FastAPI + Celery + Redis (async task queue)
  - Frontend: React 19 + Vite + TanStack Query + Tailwind CSS

---

## Project Structure

```
ai-footstep-generator/
├── src/
│   ├── pipeline/              # Core pipeline stages
│   ├── audio_backends/        # Pluggable audio generation backends
│   └── utils/                 # Shared utilities
├── tests/                     # Unit tests (pytest)
├── config/                    # Scene/prompt configuration
├── web/                       # Web interface (FastAPI + React)
├── requirements.txt
└── README.md
```

**Pluggable Backend System:**
```python
# Supports multiple audio generation backends
backend = get_backend("runpod")    # Production (serverless GPU)
backend = get_backend("mock")      # Testing (no GPU required)
# Easy to add: HuggingFace, local GPU, etc.
```

---

## Known Limitations

- **Performance varies with video quality:**
  - Best case: 80% F1 (frontal view, good lighting, clear walking)
  - Worst case: 45% F1 (side angles, occlusion, poor lighting)
- Single-person only (no multi-person tracking yet)
- Requires clear view of person walking
- Trained on walking (not running/jumping)

---

## Future Improvements

- [ ] Multi-person tracking with pose association
- [ ] Support for running/jumping gaits
- [ ] Separate training with running/walking dataset

---

## Development

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=src

# Try different backends
python -m src.main_pipeline video.mp4 --backend mock     # Fast testing
python -m src.main_pipeline video.mp4 --backend runpod   # Production quality
```

---

## Why This Project?

**Problem:** As a sound designer, creating footstep sounds for video was tedious and repetitive. Each project required manually syncing hundreds of footsteps, adjusting for different surfaces, and applying spatial audio.

**Solution:** Automate the entire process using computer vision and AI audio generation.

**What I Learned:**
- Multi-modal ML systems (vision + audio)
- Evaluation metrics don't always match perceptual quality (FAD/CLAP vs. blind testing)
- Serverless architecture significantly reduces infrastructure costs

---

## License

MIT License

---

## Contact

**Yejin Bang**
- GitHub: [@yejin-bang](https://github.com/yejin-bang)
- LinkedIn: [linkedin.com/in/yejin-bang-a61096307](https://www.linkedin.com/in/yejin-bang-a61096307/)
- Email: ybangmusic@gmail.com

---

**⭐ If you found this project useful, please consider starring it!**