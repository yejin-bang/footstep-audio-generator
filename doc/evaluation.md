# Evaluation Methodology

This document contains the **full technical evaluation details** for the Footstep Audio Generator. It is intentionally separated from the README to keep the main project overview focused on system design and outcomes, while preserving methodological rigor and reproducibility here.

---

## Evaluation Goals

The evaluation was designed to answer three questions:

1. Does fine-tuning improve **perceptual realism** of footstep audio?
2. Does the model **generalize** beyond training captions and surfaces?
3. How well do standard audio metrics correlate with professional listening judgments?

---

## Quantitative Metrics

### Fréchet Audio Distance (FAD)

**Score Range:** 20.69–32.09 across configurations

FAD measures the distance between feature distributions of generated and reference audio using embeddings from a pretrained audio classifier.

### Test Conditions

Three distinct evaluation regimes were used to probe different failure modes:

#### 1. Training Caption Test

* 1,000 generated samples
* Prompts drawn directly from training captions
* Purpose: detect overfitting or memorization

**FAD:** 24.03

#### 2. Novel Caption Test

* 1,000 generated samples
* Synthetic recombinations of surface, footwear, and context
* Purpose: evaluate linguistic generalization

**FAD:** 32.09

#### 3. Held-Out Audio Test (Primary)

* 1,236 samples (4 variations × 309 captions)
* 103 held-out audio files
* Three caption perspectives per file (material, context, character)

**FAD:** 20.69

### Reference Set Construction

* Total reference size: 1,000 audio files
* Composition:

  * 103 held-out test samples
  * 897 randomly sampled training examples

**Methodological Note:**
The reference set size is smaller than the ideal 5k–10k range recommended for stable FAD estimation. Results should therefore be interpreted comparatively rather than as absolute benchmarks.

---

### CLAP (Contrastive Language–Audio Pretraining)

**Score Range:** 0.23–0.26

CLAP evaluates semantic alignment between text prompts and generated audio in a joint embedding space.

| Test Condition    | CLAP Score |
| ----------------- | ---------- |
| Training Captions | 0.2288     |
| Novel Captions    | 0.2502     |
| Held-Out Set      | **0.2586** |

**Interpretation:**
Higher CLAP scores on held-out and novel captions suggest the model learned semantic structure rather than memorizing prompt templates.

---

## Metrics vs. Domain Reality

### Observed Disconnect

Despite moderate FAD and CLAP scores, blind listening tests consistently rated the fine-tuned model as production-usable for professional footstep work.

### Root Causes

1. **Domain Mismatch**
   Footstep audio is transient-heavy, highly structured, and short-duration — properties underrepresented in datasets used to train FAD and CLAP embedding models.

2. **Unmodeled Perceptual Dimensions**
   Professional evaluation relies on attributes such as weight, heel articulation, surface resonance, and gait rhythm — none of which are explicitly encoded in current metric spaces.

3. **Expectation-Based Listening**
   Human listeners evaluate footsteps relative to visual context (heel-to-toe timing, cadence), a multimodal dependency ignored by audio-only metrics.

**Conclusion:**
Automated metrics are useful sanity checks but insufficient for validating specialized production audio.

---

## Qualitative Evaluation: Blind Listening Tests

### Test Design

* **Total Samples:** 314 audio samples
* **Training Configurations:** 4 distinct training setups
* **Checkpoints Evaluated:** 45 total
* **Reference Model:** Stable Audio One (unfine-tuned), 12 samples
* **Blinding:** All samples shuffled and anonymized

### Evaluation Criteria

Each sample was rated on a 1–5 scale along four axes:

1. Audio Quality (artifacts, fidelity)
2. Fidelity to Professional Source Libraries
3. Prompt Adherence (surface, footwear)
4. Naturalness (temporal realism, gait perception)

### Test Coverage

* **In-distribution:** unseen surface–footwear combinations
* **Out-of-distribution:** non-footstep prompts to assess overfitting and collapse

---

## Failure Analysis & Limitations

### Running vs. Walking

* Training data combined both gaits without explicit labels
* Model defaults to running-style transients
* Walking realism suffers due to missing heel–toe–scuff patterns

### Data Imbalance

| Surface | Samples | Dataset % |
| ------- | ------- | --------- |
| Mud     | 43      | 3.0%      |
| Snow    | 38      | 2.6%      |
| Sand    | 51      | 3.5%      |

Lower sample counts correlate with reduced consistency and weaker articulation.

### Spatial Edge Cases

Extreme camera angles or partial occlusion reduce accuracy of pose-based spatialization.

---

## Takeaway

This evaluation demonstrates that:

* LoRA fine-tuning meaningfully improves perceptual realism for specialized audio
* General-purpose metrics fail to reflect domain-specific quality
* Human expert evaluation remains indispensable for production-facing generative audio systems
