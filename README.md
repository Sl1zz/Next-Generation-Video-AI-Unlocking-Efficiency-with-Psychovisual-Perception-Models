# Next Gen Video AI Unlocking Efficiency with Psychovisual Perception Models
Perception-aware video inference for computer vision. This repo explores how to cut compute and memory by ignoring redundant, non-salient frames and features, aligning models with what humans reliably perceive. Goal: faster, lighter, more robust video recognition with fewer errors from unlabeled, irrelevant stimuli.






## tl;dr
We filter non-perceptible spatiotemporal content before training/inference, guided by human contrast sensitivity and retinal receptor distribution. Results: ~15% dataset redundancy reduction and ~3.5% mean accuracy gain across diverse detectors, with faster convergence and lower resource use. 

---

## Why perception-aware?
Videos contain lots of content people **don’t actually see** at normal viewing conditions. Penalizing models for unlabeled, irrelevant stimuli adds compute and error. We apply a psychovisual weighting that emphasizes centrally perceived motion/structure while down-weighting imperceptible frequencies—yielding simpler, more “human-like” inputs. 

**Method sketch.** We estimate motion + region-of-interest, apply spatiotemporal contrast sensitivity within ROI and a peripheral falloff tied to photoreceptor density, then invert to a CV-friendly frame. (See paper for equations and pipeline diagram.)


<img width="1076" height="586" alt="image" src="https://github.com/user-attachments/assets/f685d496-8d87-4115-827a-4eae92ae4255" />

---



## Dataset
- Source: RED Komodo 6K captures → 1080p (1920×1080), 25 fps; H.264 for delivery.
- Annotations: 6,257 boxes (person 2,787; vehicle 3,470) over 1,080 frames.
- Two training sets: **Dataset 1 (unfiltered)**, **Dataset 2 (perception-filtered)**; identical labels. :contentReference[oaicite:11]{index=11}



---

## Baselines & Rationale
- **HOG+SVM (no pretraining)**
- **CNN from scratch (lightweight)**
- **YOLOv8**, **YOLOv11** (modern one-stage)
- **Faster R-CNN** (two-stage)
- **RF-DETR** (transformer)
Each baseline probes a different dependency on context/periphery; the filtered data exposes redundancies and helps most models converge faster with better mAP. 

---

## Results (high level)
Across all models, training on the **perception-filtered** set improved accuracy and reduced inference cost. Average gain ≈ **+3.5%** with faster convergence and lower memory usage; see paper figs/tables for details. 

---

## Quickstart

### 1) Environment
```bash
# Python 3.11 recommended

