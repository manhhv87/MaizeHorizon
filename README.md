# MaizeHorizon

Sensor pixels or network input: measuring the per-plant maize detection limit.

A forward-facing camera driving down a crop row turns per-plant detection into a
far-field small-object problem — a distant seedling spans a handful of pixels and
blends into the soil. Enlarging the network input is the standard reflex, and
whether it helps depends on the camera. This study locates the binding
resolution, then rules out six candidate remedies, each against a control that
would have exposed a real effect.

## Key result

Detectability is bounded by the **lesser** of the pixels the sensor resolves on
the plant and the pixels the network input carries. Which one binds is
measurable: raise the input in steps and find where half-recall stops improving.

| Rig | Nominal | Half-recall stops improving at | Binding cap |
|---|---|---|---|
| Logitech C920, Dan Phuong | 1920 | its native 1920 | sensor |
| Samsung S23 8K, Nam Sach | 7680 | ≈ 2560 | network input |

On the webcam the **sensor** binds. Half-recall sits near a 52 px native box,
pushing the input past native returns nothing on any arm, and retraining at
native 1920 gains nothing either. Equivalence across inputs is close but not
established: 15 of 24 cells pass a ±0.05 margin, 23 of 24 at ±0.075.

On the 8K camera the **input** binds instead. Raising it from 1280 to 3840 lifts
recall on a 48–64 px native plant from 0.062 to 0.419 and extends half-recall
range from 4.6 to 7.7 m on unchanged footage, saturating near 2560.

That saturation point measures the detail a camera really delivers, which need
not match its megapixel count: a nominal 7680 px frame carries nearer 2560 px of
real detail. The raw-frame power spectrum agrees, retaining 0.122 of its
low-frequency power at a matched angular frequency against 0.024 for the webcam.

Six remedies that re-use already-captured pixels fail to move the floor,
including sliced inference, which costs far-tier AP on **both** cameras (−68%
where the sensor binds, −8% and unresolved where the input binds). The one that
helps changes the learning objective rather than the pixel budget: a Normalized
Wasserstein regression term lowers the floor by 5 px, then saturates at native
resolution like everything else.

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

The first campaign also ships 28 forward-motion clips of unlabelled frames. The
second is held out entirely: never trained on, evaluation only.

Download: [doi:10.5281/zenodo.21775698](https://doi.org/10.5281/zenodo.21775698)
(CC BY 4.0). Cite that concept DOI rather than a version-specific one. Both
archives unpack into a single `data/` directory.

## Install

```bash
git clone https://github.com/manhhv87/MaizeHorizon.git
cd MaizeHorizon
pip install -r requirements.txt  # ultralytics 8.3.160 + torch

tar -xf  MaizeHorizon-images.tar
tar -xzf MaizeHorizon-annotations.tar.gz
```

Neither the checkpoints nor the result CSVs are in the repository. Train the
checkpoints first (below), then regenerate every number from them.

## Reproducing

```bash
python train_missing_seeds.py --seeds 0 1 2 3 4 --workers 2 --device 0
python reproduce_all.py                          # writes results/
```

`train_missing_seeds.py` trains only the arms that are short of seeds; per-arm
hyperparameters live in its `ARMS` table. Training entry points are `train.py`
(stock and +Mint arms), `train_tbxrd_stage2.py` (distillation), `train_nwd.py`
and `exp_multiarch_train.py`. `rerun_after_relabel.py` is the narrower path: it
regenerates only what depends on the test labels.

Two things to know before quoting any number. `max_det` differs by script: 300
for the recall and precision tables, 1000 for AP and the resolution sweep, and
**3,000 for anything on Nam Sach**, which averages 239 annotated boxes per frame
against 42 for Dan Phuong. And `n_gt` should read 4,067 = 2,747 / 1,238 / 82 for
Dan Phuong, 29,297 = 11,775 / 11,732 / 5,790 for Nam Sach.

⚠️ `--workers 2`: the default of 8 causes a fork deadlock between OpenCV and the
DataLoader on some machines — training stalls after a few epochs with the GPU at
0% and no error.

## Layout

| Path | Role |
|---|---|
| `data/` | frames, test labels, per-arm label sets, minted labels — not in the repository; created by unpacking the Zenodo archives |
| `runs/<tag>_s<seed>/` | trained weights — not in the repository; written by training |
| `results/` | every reported number, as CSV — not in the repository; written by the scripts |
| `eval_testset.py`, `eval_ap.py`, `rebuttal_common.py` | the evaluation protocol |
| `exp_*.py` | resolution sweep, equivalence, contrast, range-matched, clustered CIs |
| `train*.py`, `tbxrd_mint.py` | training and the forward-motion remedies |
| `plot_*.py`, `make_paper_figures.py`, `make_tables.py` | figures and tables |
| `scripts/` | sensor downsampling, Word build |

Each arm's `images/` is a symlink to `data/images`, so no image bytes are
duplicated.

## Citation

```bibtex
@article{maizehorizon,
  title  = {Sensor pixels or network input: measuring the per-plant maize
            detection limit},
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
