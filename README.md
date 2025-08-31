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
- Two training sets: **Dataset 1 (unfiltered)**, **Dataset 2 (perception-filtered)**
- one independent unseen test video to test the model on  

## Dataset Preview

Here are sample frames from both datasets:

### Dataset 1 – Unfiltered
| Frame 1 | Frame 2 | Frame 3 | Frame 4 | Frame 5 | Frame 6 |
|---------|---------|---------|---------|---------|---------|
| ![D1-1](data/samples/dataset1/frame1.jpg) | ![D1-2](data/samples/dataset1/frame2.jpg) | ![D1-3](data/samples/dataset1/frame3.jpg) | ![D1-4](data/samples/dataset1/frame4.jpg) | ![D1-5](data/samples/dataset1/frame5.jpg) | ![D1-6](data/samples/dataset1/frame6.jpg) |

### Dataset 2 – Perception-Filtered
| Frame 1 | Frame 2 | Frame 3 | Frame 4 | Frame 5 | Frame 6 |
|---------|---------|---------|---------|---------|---------|
| ![D2-1](data/samples/dataset2/frame1.jpg) | ![D2-2](data/samples/dataset2/frame2.jpg) | ![D2-3](data/samples/dataset2/frame3.jpg) | ![D2-4](data/samples/dataset2/frame4.jpg) | ![D2-5](data/samples/dataset2/frame5.jpg) | ![D2-6](data/samples/dataset2/frame6.jpg) |

> Note: Full datasets are hosted on [Roboflow](https://universe.roboflow.com/) – see [Dataset Access](#dataset-access).


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

