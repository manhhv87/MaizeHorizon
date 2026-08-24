# `results/dataset/` — mô tả bộ dữ liệu

## `distinct_plants.csv` — box so với cây riêng biệt

Chống lưng cho con số **988 cây riêng biệt** đứng sau 4.067 box, trích trong Phương pháp.

```bash
"$PY" exp_cluster_ci.py --labels-dir "$LAB" --images-dir "$IMG" --runs runs \
  --tags stock nearfar --seeds 0 1 2 3 4 --iou 0.3 --conf 0.001 --imgsz 1280 \
  --max-det 1000 --device 0 --out results/detection/cluster_ci.csv \
  --plants-out results/dataset/distinct_plants.csv
```

⚠️ Dòng `all` (988) **nhỏ hơn** tổng ba tầng (756+461+66 = 1.283), vì một cây đổi tầng
giữa các khung khi robot tiến lại. Đừng cộng ba tầng rồi tưởng ra tổng.

Trước ngày 2026-08-22 bảng này chỉ được **in ra màn hình** chứ không ghi CSV, nên con số
988 trong bài không truy vết được. Cờ `--plants-out` được thêm để vá đúng chỗ đó.

## `_feature_matrix.csv` — so sánh với các bộ dữ liệu khác

```bash
"$PY" dataset_feature_matrix.py \
  --out-csv results/dataset/_feature_matrix.csv \
  --out-tex paper/sections/_feature_matrix.tex
```

⚠️ Mac dinh cua script ghi ra `results/_feature_matrix.{csv,tex}` (ngoai thu muc nay),
nen hien co hai ban. Ban trong `results/dataset/` la ban dung; ban o `results/` la du.

## `image_to_clip.csv` — khung nào thuộc clip nào

Bảng tra dùng để chia tập **tách biệt theo clip**. Sinh lúc dựng dữ liệu bởi
`_archive/data_prep/match_source.py`, khớp từng khung với video nguồn bằng khoảng cách
ảnh. Script dựng dữ liệu không nằm trên đường tái lập kết quả (bộ dữ liệu công bố qua
Zenodo ở dạng đã chia), nên nó ở `_archive/`.

## `cdf.csv` — CDF diện tích box, so nhiều bộ dữ liệu

Sinh bởi `_archive/superseded/cdf_compare.py`. **Không dùng trong bản thảo hiện tại** —
hình so sánh CDF đã bị bỏ; giữ file lại phòng khi cần dựng lại. Nếu có số nào trong bài
truy về đây thì đó là lỗi, vì script sinh ra nó đã bị loại khỏi đường tái lập.
