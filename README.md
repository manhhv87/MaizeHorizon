# CornHorizon

Resolved sensor pixels limit far-field per-plant maize detection: network input size does not
extend range.

Code and evaluation protocol for a study of how far ahead a ground robot can detect individual
maize seedlings. A forward-facing camera driving down a crop row turns per-plant detection into a
far-field small-object problem: a distant seedling spans a handful of pixels and blends into the
soil. We show that detectability is governed by the pixels the sensor resolves on the plant, not by
the network input size, and that three intuitive forward-motion remedies all fail to recover the
far tier.

## Key result

Evaluating one detector at input sizes 640 to 1920 separates cause from symptom:

| Stratify recall by | Behaviour |
|---|---|
| input pixels-on-target (POT) | curves shift with input size |
| native box height (sensor pixels) | curves collapse onto one function for inputs >= 960 |

Half-recall sits near a 52 px native box height. Recall falls to near zero below about 12 px POT,
the scale at which our annotators could no longer individuate a plant either. Detection range is
therefore extended by more sensor pixels, a longer focal length or a shorter stand-off, not by a
bigger network input.

## Dataset

CornHorizon is a forward-motion, range-stratified per-plant maize benchmark: 28 clips, 5,979
frames at 1920x1080, and 3 held-out test clips hand-labelled with 4,067 plant and 941 ignore
boxes, stratified near / mid / far by pixels-on-target.

Download: Zenodo DOI, to be added.

## Install

```bash
pip install -r requirements.txt      # ultralytics 8.3.160 + torch
# unpack the Zenodo dataset into ./data
```

## Layout

```
data/
├── images/<clip>/*.jpg      5,979 frames, the only copy
├── test/{images,labels}     120 frames and their hand labels
├── arms/{stock,nearfar}/    per-arm labels, train.txt, valid.txt, data.yaml
└── mint/<clip>/             minted labels and pairs.jsonl (10,355 near/far pairs)
results/                     every number in the paper, as CSV
runs/<tag>_s<seed>/          best.pt, args.yaml, results.csv
```

Each arm's `images/` is a symlink to `data/images`, so no image bytes are duplicated. Ultralytics
derives label paths by replacing the last `/images/` with `/labels/`, so the arms can share one
image copy while keeping separate label trees.

## Reproducing

```bash
LAB=data/test/labels
IMG=data/test/images
```

`python rerun_after_relabel.py` regenerates everything below and holds the exact arguments.

| Result | Command |
|---|---|
| Per-tier AP | `eval_ap.py --labels-dir "$LAB" --images-dir "$IMG" --runs runs --tags stock nearfar distill distill_shuffle --seeds 0 1 2 --iou 0.3 --imgsz 1280 --out results/detection/testset_ap03.csv` |
| Recall ceiling | `eval_testset.py ... --iou 0.3 --conf 0.001 --out results/detection/testset_ceiling.csv` |
| Resolution sweep | `exp_resolution_sweep.py ... --tag nearfar --imgsz-list 640 960 1280 1920 --conf 0.25 --out-prefix results/scaling/scaling` |
| Cross-architecture | `exp_multiarch_eval.py ... --tags stock yolo11s yolov10s rtdetr-l --out results/baselines/rebuttal_multiarch.csv` |
| Sliced inference | `exp_sahi_baseline.py ... --tile 640 --overlap 0.2 --imgsz 1280 --out results/baselines/sahi_ap.csv` |
| Contrast terciles | `exp_contrast.py ... --out results/baselines/rebuttal_contrast` |
| Burst super-resolution | `exp_burst_sr.py --frames-dir data/images/IMG_3916_test --gt-dir "$LAB" --out results/baselines/rebuttal_burst_sr.csv` |
| Count-when-near | `furrowmap_ledger.py` then `furrowmap_count.py` |
| Figures | `plot_cliff.py`, `plot_scaling.py`, `make_paper_figures.py`, `make_teaser.py` |

Training: `train.py` (SGD, lr0 0.01, batch 8, 200 epochs, imgsz 1280) for `stock` and `nearfar`;
`train_tbxrd_stage2.py` for the distillation arms; `train_nwd.py` for NWD;
`exp_multiarch_train.py` for the other architectures; `tbxrd_mint.py` produces the minted labels.

Two things to know:

- `max_det` differs by script: 300 for the recall and precision tables, 1000 for AP and the
  resolution sweep. At conf 0.001 the cap binds, because it applies to both classes combined.
- Every CSV with an `n_gt` column should show 4,067 = 2,747 near / 1,238 mid / 82 far.

## Code

| | |
|---|---|
| `eval_testset.py`, `eval_ap.py` | evaluation protocol: POT-stratified, ignore-aware, one-to-one greedy IoU |
| `rebuttal_common.py` | shared loader and helpers; other scripts import the protocol rather than reimplementing it |
| `exp_resolution_sweep.py` | the sensor-versus-input dissociation |
| `tbxrd_mint.py`, `train_tbxrd_stage2.py`, `mve_tbd.py` | the three forward-motion remedies |
| `furrowmap_*.py` | count-when-near baseline |

## Citation

```bibtex
@article{cornhorizon,
  title  = {Resolved Sensor Pixels Limit Far-Field Per-Plant Maize Detection:
            Network Input Size Does Not Extend Range},
  year   = {2026}
}
```

## License

Code is AGPL-3.0, inherited from [Ultralytics YOLO](https://github.com/ultralytics/ultralytics),
which this work builds on. The dataset is CC BY 4.0.
