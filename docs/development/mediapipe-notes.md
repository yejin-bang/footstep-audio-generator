# Video Footstep Detection Plan

## Core Pipeline

#### Footstep_detector.py(full pipeline): 

### 1. Preprocessing & Data : pose_extractor.py

```
0. Frame Skipping (Performance Optimization)
   - Process 30fps video at 10fps (every 3rd frame)
   - 3x faster processing speed

1. Landmark Extraction (MediaPipe)
   - Hip: LEFT_HIP(23), RIGHT_HIP(24)
   - Heel: LEFT_HEEL(29), RIGHT_HEEL(30)
   - Ankle: LEFT_ANKLE(27), RIGHT_ANKLE(28)
   - Foot Index: LEFT_FOOT_INDEX(31), RIGHT_FOOT_INDEX(32)
```

### 2. Signal Processing : signal_processor.py

```
2. Cubic Spline Interpolation
   - Purpose: Fill missing values due to occlusion
   - Method: Generate smooth curves from surrounding data points

3. 10th-order Butterworth Low-pass Filter
   - Cutoff frequency: 0.1752 (research-validated value)
   - Target noise types:
     * Camera shake
     * Detection instability from lighting changes
     * Motion blur artifacts
   - Effect: Smooth signal data
```

### 3. Gait Event Detection : gait_detector.py

```
4. Hip-Foot Euclidean Distance Calculation
   - Heel Strike: Maximum Hip-Foot distance (foot extended forward)
   - Toe-off: Minimum Hip-Foot distance (foot lifting off ground)
   - Purpose: Complete gait cycle understanding

5. Peak Detection (scipy.signal.find_peaks)
   - Find local maxima in distance signal
   - Each peak = one heel strike event

4. Time Conversion -> footstep_detector.py 
   - Frame index → actual timestamps
   - Formula: timestamp = frame_index / fps 
```

## Accuracy Enhancement Techniques

### Validation & Filtering

```
7. Multi-frame Validation
   - Principle: True footsteps persist across multiple frames
   - Method: Verify consistency across consecutive frames
   - vs Butterworth: Signal smoothing vs event validation

8. Minimum Interval Setting (False Positive Prevention)
   - Global thresholds:
     * Minimum interval: 0.4s (high-speed walking)
     * General recommendation: 0.6s (99% coverage)
     * Conservative: 0.8s (nearly 100% safe)
   - Dynamic adjustment: video-specific average interval × 0.6

7. Confidence-based Filtering √
   - Exclude MediaPipe confidence < 0.7
   - Additional validation: movement distance, anatomical plausibility

10. Adaptive Thresholding (DGEI Algorithm)
    - Principle: Dynamic threshold adjustment based on individual gait patterns
    - Method:
      * Smaller steps → lower threshold (more sensitive)
      * Larger steps → higher threshold (less sensitive)
    - Formula: threshold = min_distance + (range × 0.7)
```

## Alternative & Complementary Methods

### Multi-modal Approaches

```
11. Distance + Velocity Combination
    - Distance: Hip-Foot distance-based detection
    - Velocity: Heel strike when foot velocity ≈ 0
    - Fusion: Confirmed footstep when both conditions met

12. Acceleration-based (for Vigorous Movement)
    - Principle: Rapid acceleration change upon ground contact
    - Use case: Running, stair climbing
    - vs Velocity: normal walking uses velocity, vigorous movement uses acceleration

13. Multiple Landmark Fusion
    - Top performing combinations:
      * Hip + Heel + Ankle: 97.8% accuracy (unverified)
      * Hip + Foot_Index + Toe: 96.5% accuracy
      * Hip + Knee + Ankle: 94.2% accuracy
    - Method: Weighted voting system
```

## Problem-Solving Strategies

### Occlusion Handling

```
14. Pattern Recognition-based Prediction
    A. Dynamic Time Warping (DTW)
       - Store past gait patterns as templates
       - Match templates with similar length to occluded segments
       - Performance: 98% F-measure (40m walking trials)

    B. Matrix Profile Foundation
       - Automatically discover recurring patterns in full signal
       - Find past patterns similar to pre-occlusion segment
       - Advantages: Parameter-free, batch processing compatible
```

## Technical Stack & Implementation

### Core Libraries (Validated Tools)

```python
import mediapipe as mp           # Google - Pose detection
import numpy as np               # calculation
import scipy.signal              # (Butterworth, find_peaks)
import cv2                       # video processor
from scipy.interpolate import interp1d  # Cubic spline

# For Occlusion (optional)
# DTW: pip install dtaidistance
# Matrix Profile: pip install stumpy
```

### Performance Optimization (Batch Processing)

```
- Frame skipping: 30fps → 10fps (3x faster)
- Vectorized operations: NumPy for efficient computation
- Memory efficient: Load full video then batch process
- Parallel processing: Multiple videos simultaneously
```

## Validation & Testing

### Performance Metrics

```
- Minimum target: 70% accuracy (initial implementation)
- Practical target: 85% accuracy (production-ready)
- Research target: 95%+ accuracy (paper-level)

- Processing speed: 1-minute video → <30 seconds processing
- False Positive Rate: <10%
- Miss Rate: <15%
```

### Test Scenarios

```
1. Basic walking (indoor, good lighting)
2. Fast walking and running
3. Occlusion situations (behind people/objects)
4. Low-light environments
5. Various ages and body types
```

## Development Phases

### Phase 1: MVP (2 weeks)

```
- MediaPipe + Hip-Foot distance only
- Basic peak detection
- Target: 70% accuracy
```

### Phase 2: Enhancement (3 weeks)

```
- Add signal processing (interpolation, filtering)
- Multi-frame validation
- Target: 85% accuracy
```

### Phase 3: Optimization (4 weeks)

```
- Multiple landmark fusion
- Occlusion handling
- Target: 95% accuracy
```

---

**Core Principles**

- Prioritize validated libraries
- Incremental improvement (working over perfect)
- User experience-focused development
- Balance performance and accuracy

