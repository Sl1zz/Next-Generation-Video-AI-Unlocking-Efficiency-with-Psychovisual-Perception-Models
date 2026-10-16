# PSNR-VR: A Full-Reference Quality Metric for Wide-Angle VR Video

**Mean opinion scores, objective metric outputs and correlation results** accompanying the paper:

> S. J. Rogers, T. Weti, G. A. Voronkov, P. Delmas, and A. Mozhaeva, "PSNR-VR: A Full-Reference Quality Metric for Wide-Angle VR Video Incorporating Human Visual System Models," *under review*, 2026.

PSNR-VR keeps the simple PSNR formulation but weights pixel errors by how visible they are to a human viewer. It uses models of spatial, temporal and peripheral visual sensitivity, centred on a region of interest (ROI). This repository contains the data needed to check and reproduce the paper's evaluation. It does not include the metric implementation.

---

## At a Glance

| | |
|---|---|
| **Content** | 7 wide-angle (360°) source videos from LIVE-FBT-FCVR, 7680 × 3840 |
| **Distortions** | Foveated VP9 compression at CRF 51 (higher quality) and CRF 63 (lower quality) |
| **Test sequences** | 14 (7 contents × 2 levels) |
| **Observers** | 20 participants, aged 20–40 |
| **Display** | HTC Vive Pro 2 (2448 × 2448 per eye, 120 Hz, 120° FoV) |
| **Method** | ITU-R BT.500 single stimulus continuous quality (SSCQ), 0–5 scale |
| **Metrics compared** | PSNR-VR, PSNR-sp-VR, PSNR, SSIM, BRISQUE, FovVideoVDP |

### Key result

| Metric | PLCC | PLCC 95% CI | SROCC | p vs PSNR-VR (Williams) |
|---|---|---|---|---|
| **PSNR-VR** | **0.939** | [0.81, 0.98] | **0.894** | – |
| FovVideoVDP | 0.935 | [0.80, 0.98] | 0.855 | 0.903 (not significant) |
| PSNR-sp-VR | 0.875 | [0.64, 0.96] | 0.788 | 0.082 |
| SSIM | 0.811 | [0.49, 0.94] | 0.866 | 0.042 |
| PSNR | 0.662 | [0.20, 0.88] | 0.684 | 0.006 |
| BRISQUE | 0.587 | [0.08, 0.85] | 0.488 | 0.013 |

PSNR-VR correlates significantly better with MOS than PSNR, SSIM and BRISQUE (two-tailed Williams test, α = 0.05). It is statistically equivalent to FovVideoVDP while using a much simpler closed-form formulation, at roughly one third of FovVideoVDP's per-frame CPU processing time in our implementation.

---

## Repository Contents

```
data/
├── metric_scores.csv        # All metric outputs + MOS for the 14 sequences (paper Table I)
└── correlations.csv         # PLCC / SROCC, 95% CIs, Williams-test p-values, within-CRF results (paper Table II)
LICENSE
README.md
```

### `metric_scores.csv`

| Column | Description |
|---|---|
| `content` | Source video name (LIVE-FBT-FCVR) |
| `crf` | VP9 constant rate factor (51 or 63) |
| `psnr_vr_db` | PSNR-VR, dB (higher is better) |
| `psnr_db` | Standard luma PSNR, dB |
| `ssim` | SSIM, 0–1 |
| `fovvideovdp_jod` | FovVideoVDP, JOD units (higher is better) |
| `brisque` | BRISQUE, 0–100 (**lower is better**; sign reversed for correlation) |
| `psnr_sp_vr_db` | PSNR-sp-VR (no temporal weighting), dB |
| `mos` | Mean opinion score, 0 (Bad) – 5 (Excellent) |

---

## Test Content

| Content | Scene |
|---|---|
| Coco | Market square |
| EER | Shopping centre |
| Microsoft | Inside store |
| Oasis | Restaurant |
| Sculpture | Riverbank |
| Stadium | Road view |
| Waterfall | Park |

Three further LIVE-FBT-FCVR contents (Playground, SanJacinto and Starbucks) were excluded from the metric evaluation; see Section IV-A and the Discussion of the paper.

The source and compressed videos belong to the **LIVE-FBT-FCVR** database and are **not redistributed here**. Request them from the University of Texas at Austin: <http://live.ece.utexas.edu/research/LIVEFBTFCVR/>

---

## Subjective Experiment

The ratings were collected in our earlier controlled VR study (Mozhaeva et al., IVCNZ 2025) and are reused here for the first time to evaluate objective metrics.

- Sequences (10 s each) were shown in random order on a Vive Pro 2.
- Participants were instructed to look at a fixed point marked at the centre of the region of interest defined in Jin et al. (SPIE 2020).
- After each sequence, a rating bar with five anchors (Bad, Poor, Fair, Good, Excellent) appeared, and participants selected a continuous value from 0 to 5.
- Head orientation was tracked by the HMD. Trials with an average deviation from the fixation point above 5% per frame were removed (less than 4% of observations).
- MOS is the mean of the retained ratings, reported with its 95% confidence interval following ITU-R BT.500.

**Ethics:** The study was approved by the Research Ethics Approvals Committee of the Eastern Institute of Technology (LR18140725). All participants gave written informed consent.

---

## Evaluation Protocol

- **PLCC** measures linear agreement with MOS. **SROCC** measures whether the metric ranks sequences in the same order as viewers.
- 95% confidence intervals for PLCC use the Fisher *z*-transformation (n = 14).
- PLCC differences are tested with the two-tailed Williams test for dependent correlations (Steiger, 1980) at α = 0.05, over all 14 sequences and within each compression level (n = 7).
- BRISQUE scores are sign-reversed before correlation so that higher values always mean better quality.
- Frame-based metrics are averaged over each sequence (PSNR-VR: mean of per-frame dB values). FovVideoVDP produces a sequence-level score.

---

## Related Work

This dataset builds on our earlier study of perceptually optimised wide-angle VR transmission:

> A. Mozhaeva, N. Alani, I. Vlasuyk, S. J. Rogers, A. Potashnikov, and L. Streeter, "Enhancing wide-angle VR video transmission using human perception," *Proc. IVCNZ*, 2025, pp. 267–272. doi: [10.1109/IVCNZ67716.2025.11281650](https://doi.org/10.1109/IVCNZ67716.2025.11281650)

---

## Citation

If you use this data, please cite:

```bibtex
@inproceedings{rogers2026psnrvr,
  author    = {Rogers, Steven J. and Weti, Tainui and Voronkov, Gleb A. and Delmas, Patrice and Mozhaeva, Anastasia},
  title     = {{PSNR-VR}: A Full-Reference Quality Metric for Wide-Angle {VR} Video
               Incorporating Human Visual System Models},
  year      = {2026},
  note      = {Under review}
}
```

Please also cite the LIVE-FBT-FCVR database:

```bibtex
@article{jin2021livefbtfcvr,
  author  = {Jin, Yize and Chen, Meixu and Goodall, Todd and Patney, Anjul and Bovik, Alan C.},
  title   = {Subjective and Objective Quality Assessment of {2D} and {3D} Foveated Video
             Compression in Virtual Reality},
  journal = {IEEE Trans. Image Process.},
  volume  = {30},
  pages   = {5905--5919},
  year    = {2021},
  doi     = {10.1109/TIP.2021.3087322}
}
```

---

## License

Data in this repository is released under **[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)**. LIVE-FBT-FCVR content remains subject to its original licence.

## Contact

Steven J. Rogers, Eastern Institute of Technology, <srogers@eit.ac.nz>
