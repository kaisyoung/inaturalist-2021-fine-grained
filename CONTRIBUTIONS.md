# Contributions

A five-person group project in computer vision: fine-grained species
classification using a 500-class subset of iNaturalist-2021
(20,000 train / 5,000 validation / 5,000 test).

The project compares a classical computer-vision baseline against fine-tuned
CNNs, with a corruption-robustness study and Grad-CAM explainability.

---

## Kaishi Yang — classical computer vision

I owned the handcrafted-feature baseline: design, implementation, experiments,
and the corresponding sections of the report.

### What I built

A block-structured descriptor and the selection experiments behind it.

| Block | Dimensions | Content |
|---|---:|---|
| Colour | 520 | joint hue–saturation histogram (12×8) + 8-bin value histogram, over a two-level spatial pyramid |
| Texture | 140 | multi-scale uniform LBP, `(P=8,R=1)` and `(P=16,R=2)`, same pyramid |
| HOG | 1,764 | 9 orientations, 16×16 cells, 2×2 blocks, L2-Hys |
| **Handcrafted total** | **2,424** | |
| SIFT bag-of-visual-words | 1,280 | 256-word vocabulary, top-3 soft assignment with Gaussian weights, 1 + 2×2 spatial pyramid |
| **Combined** | **3,704** | |

Normalisation is a fitted step of the estimator pipeline rather than part of
extraction: each block is L1-normalised, raised to a power `α`, then
L2-normalised. At `α=0.5` this is the Hellinger map. Keeping it in the pipeline
means a cached descriptor can be re-normalised for an ablation without
re-reading any image, and per-image inference on corrupted images automatically
applies the transform used at training.

Model selection ran a 19-configuration validation grid over the exponent,
block-wise versus global normalisation, feature subsets, and classifier choice.

### Result

The implementation replaced an earlier handcrafted baseline. On the identical
split and test set:

| Metric | Replaced implementation | This implementation |
|---|---:|---:|
| Test top-1 | 0.0410 | **0.1252** |
| Test top-5 | 0.1106 | **0.2678** |
| Test macro-F1 | 0.0339 | **0.1116** |

Roughly 3× on both top-1 and macro-F1.

### Findings worth recording

Two results overturned conclusions I had been about to write up.

**A cheaper surrogate solver inverted the ranking it was being used to
establish.** Averaged SGD is about 30× faster than liblinear across 500
one-vs-rest problems, so I used it to screen ablations. It produced the
*opposite* ordering from liblinear on both feature-block composition and
normalisation assembly. Three conclusions drawn from the screening pass did not
survive re-running on the real solver.

**Leave-one-out ablations describe one particular feature set, not a block in
isolation.** Without the visual-word block, dropping HOG or texture *improved*
validation accuracy; with it, the same two blocks contributed +1.63 points
together. A block's measured contribution depends on what else is in the vector.

The selected exponent was also an interior optimum at `α=0.3` rather than the
theoretically clean `α=0.5`, so the exponent has to be searched rather than
assumed.

### Files

Modules I designed and implemented, under `src/traditional/`:

| File | Purpose |
|---|---|
| `features.py` | colour, texture and HOG blocks; the per-image entry point used by the robustness study |
| `normalize.py` | `PowerNormalizer`, the block-wise L1 → power → L2 pipeline step |
| `bovw.py` | SIFT extraction, vocabulary fitting, soft assignment, spatial pooling |
| `extract_features.py` | handcrafted feature cache |
| `extract_bovw_features.py` | visual-word feature cache |
| `train_traditional.py` | training and the ablation grid |
| `evaluate_traditional.py` | validation and held-out test evaluation |
| `freeze_stage2.py` | the Stage 2 record consumed downstream |

Also `scripts/rehash_stage2_evidence.py`, which re-anchors the hard-coded Stage 2
artifact hashes that the downstream gates check.

Integrating the replacement required changes to five files owned by other team
members — the robustness evaluator's hard-coded descriptor width, the Stage 2
aggregator's schema and dimension checks, two report generators' labels, and the
citation whitelist. Those changes were forced by the replacement rather than
incidental.

### Note on the submitted version

My implementation was 1,750 lines across the nine modules above. Teammates later
expanded it to 6,065 lines by adding an engineering layer — atomic writes, cache
row-identity checks, pilot gates, parameter validation. I verified by AST
comparison that the algorithm itself is unchanged: every difference in
`features.py` and `normalize.py` is a rename, a docstring rewrite, or added input
validation. One substantive change was made to `bovw.py`, replacing the
`np.argpartition` top-k search with `cv2.batchDistance`; the Gaussian weighting
is mathematically identical, and the difference it produces is about 0.5 points.

### Version control

In the team's original repository, `src/` was never committed to `main`: that
default branch holds `scripts/`, `tests/`, `configs/`, `docs/` and `report/`, but
none of the 48 source files, because the initial code was added through GitHub's
web upload. The `traditional-rewrite` branch there is where the source tree first
entered version control. Six commits on it are mine:

```
da853e2  2026-07-31  Add .gitignore for data and large intermediates
6c8cebd  2026-07-31  Reimplement the traditional method and track src/ in git
cd6be1e  2026-08-02  Add rerun notes for the traditional-method replacement
59a10c8  2026-08-02  Add a script to re-anchor the pinned Stage 2 evidence hashes
54dd82a  2026-08-03  Bind the robustness baseline to handcrafted features only
4eae38b  2026-08-03  Correct the rerun instructions: Stage 4 must be re-run in full
```

Those commits also brought the rest of the team's existing source tree under
version control for the first time. Committing a file is not authorship of it:
the deep-learning, explainability and analysis modules are other members' work.

---

## Other members

The deep-learning track (ResNet18 from scratch and pretrained, EfficientNet-B0
pretrained, the augmentation and frozen-head ablations), the Grad-CAM
explainability study, the corruption-robustness protocol, the result aggregation
and report-generation pipeline, and the dataset extraction and split preparation
were built by other members of the group.
