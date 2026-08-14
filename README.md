# MaizeHorizon

Far-field per-plant maize detection: network input size does not lower the floor.

A forward-facing camera driving down a crop row turns per-plant detection into a
far-field small-object problem — a distant seedling spans a handful of pixels and
blends into the soil. Enlarging the network input is the standard reflex; this
study shows it does not lower the floor, then rules out six candidate remedies,
each against a control that would have exposed a real effect.

## Key result

Stratifying recall by **input pixels-on-target** makes the curves shift with input
size; stratifying by **native box height** makes them equivalent across inputs
≥ 960, within ±0.073 recall. Across 24 paired comparisons among inputs 960, 1280
and 1920, every difference falls inside that band and 12 favour the larger input
against 12 the smaller — no systematic gain.

Half-recall sits near a 52 px native box height at the deployed operating point
(confidence 0.25), but that height is not a sensor constant: it moves with
plant–soil contrast (48–57 px) and drops to about 29 px at the recall ceiling.

## Dataset

A forward-motion, range-stratified per-plant maize benchmark: 28 clips, 5,979
frames at 1920×1080, with 3 held-out test clips hand-labelled with 4,067 plant and
941 ignore boxes, stratified near/mid/far by pixels-on-target.

Download: [doi:10.5281/zenodo.21775698](https://doi.org/10.5281/zenodo.21775698)
(CC BY 4.0). Cite that concept DOI rather than a version-specific one. Both
archives unpack into a single `data/` directory.

## Install

```bash
git lfs install                  # the checkpoints are stored via Git LFS
git clone https://github.com/manhhv87/MaizeHorizon.git
cd MaizeHorizon
pip install -r requirements.txt  # ultralytics 8.3.160 + torch

tar -xf  MaizeHorizon-images.tar
tar -xzf MaizeHorizon-annotations.tar.gz
```

The 36 trained checkpoints (12 arms × 3 seeds, 916 MB) sit in `runs/` via Git LFS,
so every number in `results/` can be checked without retraining. Cloning without
Git LFS leaves the weights as pointer files.

## Reproducing

`python rerun_after_relabel.py` regenerates every table in `results/` and holds
the exact arguments for each step. Training entry points are `train.py` (stock and
nearfar arms), `train_tbxrd_stage2.py` (distillation), `train_nwd.py`, and
`exp_multiarch_train.py`.

Two things to know: `max_det` differs by script (300 for recall and precision
tables, 1000 for AP and the resolution sweep), and every CSV with an `n_gt` column
should show 4,067 = 2,747 near / 1,238 mid / 82 far.

## Layout

| Path | Role |
|---|---|
| `data/` | frames, test labels, per-arm label sets, minted labels — not in the repository; created by unpacking the Zenodo archives |
| `results/` | every reported number, as CSV |
| `runs/<tag>_s<seed>/` | `weights/best.pt` (Git LFS), `args.yaml`, `results.csv` |
| `eval_testset.py`, `eval_ap.py`, `rebuttal_common.py` | the evaluation protocol |
| `exp_*.py` | resolution sweep, equivalence, contrast, range-matched, clustered CIs |
| `train*.py`, `tbxrd_mint.py` | training and the forward-motion remedies |
| `plot_*.py`, `make_paper_figures.py` | figures |

Each arm's `images/` is a symlink to `data/images`, so no image bytes are
duplicated.

## Citation

```bibtex
@article{maizehorizon,
  title  = {Far-Field Per-Plant Maize Detection: Network Input Size
            Does Not Lower the Floor},
  author = {Hoang, Manh V.},
  year   = {2026}
}

@dataset{maizehorizon_data,
  title     = {MaizeHorizon: a forward-motion, range-stratified per-plant maize
               detection dataset},
  author    = {Hoang, Manh V.},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21775698}
}
```

## License

Code is AGPL-3.0, inherited from
[Ultralytics YOLO](https://github.com/ultralytics/ultralytics). The dataset is
CC BY 4.0.
