# Footstep Audio Generator

**Automated footstep sound effect generation and synchronization for film and game production**

An end-to-end system that generates realistic footstep sound effects from video, automatically synchronized with character movement and spatially processed. Designed and built to solve one of the most time-consuming tasks in audio post-production: creating believable, frame-accurate footsteps at scale.

[Demo Video](https://youtu.be/2v67SE0sxHg)

---

## What It Does

* Detects footsteps and walking patterns directly from video
* Generates context-aware footstep sounds (surface, footwear, character)
* Synchronizes audio to visual contact points frame-accurately
* Applies spatial processing (panning and attenuation based on screen position)
* Outputs production-ready audio files (WAV or embedded in video)

**Tech Stack:** Python, PyTorch, Stable Audio Tools, LoRA, CLIP, MediaPipe, OpenCV, Docker

---

## System Overview

The pipeline combines computer vision, scene understanding, and generative audio into a single automated workflow:

| Stage | Component          | Technology              | Output                                               |
| ----- | ------------------ | ----------------------- | ---------------------------------------------------- |
| 1     | Video Validation   | OpenCV                  | Duration, FPS, resolution                            |
| 2     | Footstep Detection | MediaPipe Pose          | Foot contact timestamps, left/right, screen position |
| 3     | Scene Analysis     | CLIP (ViT-B/32)         | Environment classification → audio prompt            |
| 4     | Audio Generation   | LoRA-tuned Stable Audio | Multiple footstep variations                         |
| 5     | Spatial Processing | Constant-power panning  | Screen-position-aware audio                          |
| 6     | Finalization       | Mixing & normalization  | Production-ready WAV / video                         |

---

## Model & Training Approach

### Foundation

* **Base Model:** Stable Audio One
* **Fine-Tuning:** LoRA-based adaptation focused exclusively on footstep audio
* **Frameworks:** PyTorch 2.5.0, Librosa, NumPy

The goal of training was not general audio quality, but **perceptual realism under professional sound design standards**.

### Training Decisions (High-Level)

* Evaluated multiple optimizers and learning-rate schedules; retained only configurations that directly addressed observed training bottlenecks
* Implemented LoRA-specific gradient monitoring to avoid misleading signals from frozen base-model parameters
* Separated model configuration from training configuration to enforce clearer ownership and reproducibility

Detailed training experiments and comparisons are documented separately in the evaluation materials.

---

## Training Data

**Source:** Professional sound design asset library used in commercial film projects

* ~2.4 hours of audio (1,444 segments, 6 seconds each)
* 32 surface–footwear combinations
* Surfaces include: concrete, dirt, gravel, leaves, marble, metal, sand, snow, water, wood, and more

### Data Preparation

* Segmented at silent regions to preserve transient integrity
* Enforced stereo consistency and loudness normalization (-16 LUFS)
* Applied fades and generated multi-perspective captions per segment

**Augmentation:** Tested and intentionally rejected — even minimal augmentation caused perceptible degradation unacceptable for professional use.

---

## Results & Evaluation (Summary)

### Qualitative Performance

The fine-tuned model consistently outperforms the unfine-tuned base model for footstep generation, particularly in:

* Heel articulation and transient clarity
* Perceived weight and material resonance
* Dress shoes, boots, and barefoot realism

Output quality is suitable for direct production use in well-represented surface–footwear combinations.

### Metrics vs. Reality

Standard audio metrics (FAD, CLAP) showed limited correlation with perceived quality. Blind listening tests by a professional sound designer revealed strong realism and contextual accuracy, highlighting a key insight:

> **For specialized audio domains, human expert evaluation remains essential and cannot be replaced by generic automated metrics.**

A detailed breakdown of quantitative and qualitative evaluation is available in [`evaluation.md`](./doc/evaluation.md).

---

## Known Limitations

* **Running vs. Walking:** The model tends to generate running-style footsteps due to mixed gait data in training
* **Rare Surfaces:** Underrepresented surfaces (e.g., mud, snow) show reduced consistency
* **Edge-Case Spatial Accuracy:** Extreme camera angles can affect spatialization

These limitations are well-scoped and directly inform planned dataset and conditioning improvements.

---

## Deployment & Usage

### Quick Start (Docker Compose)

```bash
docker-compose up
```

Launches:

* React frontend (`localhost:5173`)
* FastAPI backend (`localhost:8000`)
* Celery worker and Redis queue

The interface supports video upload, progress monitoring, footstep preview, and audio download.

**Note:** The fine-tuned model weights are not included due to size constraints; this repository demonstrates the full system architecture.

---

## About

Created by **Yejin Bang** — professional sound designer and film composer transitioning into ML engineering. This project combines domain expertise in audio production with modern machine learning systems design.

**Contact:** [ybangmusic@gmail.com](mailto:ybangmusic@gmail.com)

---

## License

MIT License
