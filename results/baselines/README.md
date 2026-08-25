# `results/baselines/` — các baseline "thêm pixel" và phân tầng theo tương phản

```bash
PY=/home/manhhv/anaconda3/envs/MAAF-NET/bin/python
LAB="data/test/labels";  IMG="data/test/images"
```

## `sahi_ap.csv` / `sahi_ap_n5.csv` — phát hiện theo ô cắt

```bash
"$PY" exp_sahi_baseline.py --labels-dir "$LAB" --images-dir "$IMG" --runs runs \
  --tag nearfar --seeds 0 1 2 3 4 --tile 640 --overlap 0.2 --imgsz 1280 \
  --iou 0.3 --conf 0.001 --device 0 \
  --out results/baselines/sahi_ap_n5.csv \
  --per-seed-out results/baselines/sahi_perseed_1080p_n5.csv
```

`--per-seed-out` ghi AP thô từng seed. Bản tổng hợp chỉ có mean/std, không đủ để chạy
Welch hay phép kiểm tương tác — xem `results/rebuttal/README.md`.

Nửa 8K của cùng phép so nằm ở `results/crosssite/README.md`.

## `rebuttal_multiarch.csv` — sàn phát hiện qua bốn kiến trúc

```bash
"$PY" exp_multiarch_eval.py --labels-dir "$LAB" --images-dir "$IMG" --runs runs \
  --tags stock yolo11s yolov10s rtdetr-l --seeds 0 1 2 3 4 --imgsz 1280 \
  --iou 0.3 --device 0 --out results/baselines/rebuttal_multiarch.csv
```

Dòng YOLOv8-s dùng tag `stock`; không có `runs/yolov8s_s*` riêng.

## `rebuttal_contrast_*.csv` — phân tầng theo tương phản cây–đất

`exp_contrast.py` nhan **danh sach .pt** qua `--weights`, khong nhan `--runs/--tag/--seeds`:

```bash
W=$(ls runs/nearfar_s{0,1,2,3,4}/weights/best.pt | paste -sd,)
"$PY" exp_contrast.py --labels-dir "$LAB" --images-dir "$IMG" --weights "$W" \
  --imgsz 1280 --device 0 --out results/baselines/rebuttal_contrast
```

→ `rebuttal_contrast_by_contrast.csv` và `rebuttal_contrast_h50.csv` (script nối hậu tố).
Tương phản = excess green trung bình trong box trừ trung bình trên vành bao quanh rộng
bằng một bề ngang box, đã che các cây khác; chia ba nhóm tại phân vị 33,3 và 66,7.

## `rebuttal_burst_sr_*.csv` — siêu phân giải theo chuỗi ảnh

Một file cho mỗi clip. Đối chứng: phóng bicubic (không mang thông tin đa khung) và chồng
ảnh xáo trộn (đúng bộ máy canh ảnh, sai nội dung).

Cung vay, `exp_burst_sr.py` nhan `--weights` va `--gt-dir` (khong phai `--labels-dir`):

```bash
W=$(ls runs/nearfar_s{0,1,2,3,4}/weights/best.pt | paste -sd,)
for CLIP in IMG_3916 IMG_3936; do
  "$PY" exp_burst_sr.py --weights "$W" --frames-dir "data/images/$CLIP" \
    --gt-dir "$LAB" --imgsz 1280 --device 0 \
    --out "results/baselines/rebuttal_burst_sr_$CLIP.csv"
done
```

→ `rebuttal_burst_sr_IMG_3916.csv` va `rebuttal_burst_sr_IMG_3936.csv`.

⚠️ `rerun_after_relabel.py` **bo qua** buoc nay tru khi dat bien `BURST` o dau file
(mac dinh `None`), vi no can duong dan toi thu muc khung lien tiep cua tung clip.
Kiem `--frames-dir` khop bo cuc thu muc anh hien tai truoc khi chay.

## `sahi_ap_n5_fpfilt.csv` — bản chạy lại theo chuẩn COCO

Sau khi `eval_ap.py` chuyển sang đúng chuẩn COCO (lọc kích thước cho false positive +
tích phân 101 điểm), mọi AP theo tầng đổi số, nên nửa 1080p của phép đảo dấu phải chạy lại.
Cờ `--fp-size-filter` bật đúng luật COCO: dự đoán không khớp mà **nằm ngoài** dải kích
thước của tầng thì bị bỏ qua, không tính là false positive. Thiếu luật này thì AP tầng xa
đếm cả những dự đoán cỡ gần — đo được 88,6% false positive tầng xa là box cỡ near.

```bash
"$PY" exp_sahi_baseline.py --labels-dir "$LAB" --images-dir "$IMG" --runs runs \
  --tag nearfar --seeds 0 1 2 3 4 --tile 640 --overlap 0.2 --imgsz 1280 \
  --iou 0.3 --conf 0.001 --device 0 --fp-size-filter \
  --out results/baselines/sahi_ap_n5_fpfilt.csv \
  --per-seed-out results/baselines/sahi_perseed_1080p_n5_fpfilt.csv
```

Hiệu ứng lớn hơn hẳn bản cũ: tầng xa 0,299 → 0,092 (−69%), so với chênh lệch nhỏ trước đây.
