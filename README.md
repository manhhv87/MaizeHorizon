# MaizeHorizon

Sensor resolution and network input size jointly bound far-field per-plant maize
detection.

A forward-facing camera driving down a crop row turns per-plant detection into a
far-field small-object problem — a distant seedling spans a handful of pixels and
blends into the soil. Enlarging the network input is the standard reflex, and
whether it helps depends on the camera. This study locates the binding
resolution, then rules out six candidate remedies, each against a control that
would have exposed a real effect.

## Key result

Detectability is bounded by the **lesser** of the pixels the sensor resolves on
the plant and the pixels the network input carries. Which one binds is measurable:
raise the input in steps and find where half-recall stops improving.

| Rig | Nominal | Saturates at | Effective resolution |
|---|---|---|---|
| Logitech C920, Dan Phuong | 1920 | input ≈ 960 | ≤ 960–1280 |
| Samsung S23 8K, Nam Sach | 7680 | input ≈ 2560–3840 | ≈ 2560–3840 |

On the webcam the **sensor** binds: recall by native box height is equivalent
across inputs ≥ 960 within ±0.073, half-recall near a 52 px native box, and
retraining at native 1920 gains nothing. On the 8K camera the **input** binds
instead — raising it lifts recall on a 60 px native plant from 0.088 to 0.496 and
extends half-recall range from 4.6 to 7.8 m on unchanged footage.

The saturation point measures the detail a camera really delivers, which need not
match its megapixel count. Two further measurements agree that the 8K frames
carry nearer 2560 px of real detail: the power spectrum of the raw frames, and
the sign of sliced inference, which loses 85% far AP where the sensor is
exhausted and gains 143% where it is not.

## Dataset

A forward-motion, range-stratified per-plant maize benchmark spanning two
campaigns that differ in locality, season and camera:

| | Dan Phuong, Hanoi | Nam Sach, Hai Phong |
|---|---|---|
| Season | April 2025 | July 2026 |
| Camera | Logitech C920, 1920×1080 | Samsung Galaxy S23, 7680×4320 |
| Labelled frames | 120 (3 held-out clips) | 150 (2 sessions) |
| Plant / ignore boxes | 4,067 / 941 | 29,297 / 6,482 |
| Far tier at input 1280 | 82 | 5,790 |
| … of which sub-12 px POT | 4 | 2,324 |

The first also ships 28 clips and 5,979 unlabelled frames. The second is held out
entirely: never trained on, evaluation only.

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

Two things to know. `max_det` differs by script: 300 for recall and precision
tables, 1000 for AP and the resolution sweep, and **3,000 for anything on Nam
Sach**, which averages 238 annotated plants per frame against 34 for Dan Phuong.
And `n_gt` should read 4,067 = 2,747 / 1,238 / 82 for Dan Phuong, 29,297 =
11,775 / 11,732 / 5,790 for Nam Sach.

`results/crosssite/README.md` holds the cross-site commands and four
interpretation traps worth reading before quoting any number from them.

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
| `exp_range_validation.py`, `scripts/` | metric-distance validation, sensor ablation, Word build |

Each arm's `images/` is a symlink to `data/images`, so no image bytes are
duplicated.

## Citation

```bibtex
@article{maizehorizon,
  title  = {Sensor resolution and network input size jointly bound
            far-field per-plant maize detection},
  author = {Hoang, Manh V. and Nguyen, Truong Q.},
  year   = {2026}
}

@dataset{maizehorizon_data,
  title     = {MaizeHorizon: a forward-motion, range-stratified per-plant maize
               detection dataset},
  author    = {Hoang, Manh V. and Nguyen, Truong Q.},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21775698}
}
```

## License

Code is AGPL-3.0, inherited from
[Ultralytics YOLO](https://github.com/ultralytics/ultralytics). The dataset is
CC BY 4.0.
