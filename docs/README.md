# Documentation Index

Welcome to the Footstep Audio Generation Pipeline documentation!

## 📚 Core Documentation

### Getting Started
- **[Setup Guide](SETUP.md)** - Installation and environment setup
- **[Pipeline Overview](PIPELINE.md)** - Complete pipeline architecture and flow

### User Guides
- **[Single Audio Generation](guides/single-audio-generation.md)** - Generate individual audio clips

## 🔧 API Documentation

- **[Scene Analyzer API](api/scene-analyzer.md)** - CLIP-based scene analysis module

## 💻 Development Documentation

### Project Organization
- **[Reorganization Summary](development/REORGANIZATION.md)** - How the project was restructured
- **[Improvements Log](development/IMPROVEMENTS.md)** - Previous fixes and enhancements

### Technical Notes
- **[MediaPipe Notes](development/mediapipe-notes.md)** - Pose estimation implementation details
- **[Stable Audio Notes](development/stable-audio-notes.md)** - Audio generation model notes

### Experiments
- **[Raw vs Signal Processing Analysis](experiments/raw_vs_signal_processing_analysis_09232025.md)** - Detection algorithm comparison

## 🧪 Testing

See [tests/README.md](../tests/README.md) for unit testing documentation.

## 📖 Quick Links

- **[Main README](../README.md)** - Project overview and quick start
- **[Tests README](../tests/README.md)** - How to run tests
- **[Config Files](../config/)** - Configuration file reference

---

## Documentation Structure

```
docs/
├── README.md                  # This file - documentation index
├── SETUP.md                   # Setup and installation guide
├── PIPELINE.md                # Pipeline architecture documentation
├── api/                       # API documentation
│   └── scene-analyzer.md
├── guides/                    # User guides and how-tos
│   └── single-audio-generation.md
├── development/               # Development documentation
│   ├── REORGANIZATION.md      # Project reorganization summary
│   ├── IMPROVEMENTS.md        # Previous improvements
│   ├── mediapipe-notes.md     # MediaPipe technical notes
│   └── stable-audio-notes.md  # Stable Audio technical notes
└── experiments/               # Experimental analysis and results
    └── raw_vs_signal_processing_analysis_09232025.md
```

---

**Last Updated**: 2025-11-18
