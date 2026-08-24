# `results/rebuttal/` — các phép đo bổ sung cho phản biện

Mỗi mục ghi **lệnh đã chạy** để sinh ra CSV tương ứng. Các biến đường dẫn:

```bash
PY=/home/manhhv/anaconda3/envs/MAAF-NET/bin/python
LAB="data/test/labels";  IMG="data/test/images"                       # bộ Đan Phượng, 1080p
LAB2=/media/manhhv/DATA/AI/_archive/paper2/test2/labels                # bộ Hải Phòng, 8K
IMG2=/media/manhhv/DATA/AI/_archive/paper2/test2/images
```

---

## M2 — khớp trần phát hiện giữa hai bộ

Bộ Đan Phượng chấm ở `max_det=300`, bộ 8K ở `3000`. Không khớp trần thì không so được
hai chiều cao nửa recall. Chấm lại bộ thứ nhất ở trần lớn hơn:

```bash
"$PY" eval_testset.py --labels-dir "$LAB" --images-dir "$IMG" --runs runs \
  --tags stock nearfar --seeds 0 1 2 --iou 0.3 --conf 0.001 --imgsz 1280 \
  --max-det 3000 --device 0 --out results/rebuttal/M2_danphuong_md3000.csv
```

→ `M2_danphuong_md3000.csv`, `M2_danphuong_md3000_curve.csv` (script tự thêm `_curve`),
`M2_h50_matched_cap.csv` (h50 tính lại từ đường cong đã khớp trần).

Kết quả: chỉ nhánh `stock` nhạy với trần (far recall +0,077); nhánh tự gán nhãn dịch 0,004.

## M3 — quét độ phân giải cho RT-DETR-l

Sàn phát hiện có bất biến theo kiến trúc không, và **mức** của nó có dịch không:

```bash
"$PY" exp_resolution_sweep.py --labels-dir "$LAB" --images-dir "$IMG" --runs runs \
  --tag rtdetr-l --seeds 0 1 2 --imgsz-list 640 960 1280 1920 \
  --iou 0.3 --conf 0.25 --device 0 --out-prefix results/rebuttal/M3_rtdetr_sweep
```

→ `M3_rtdetr_sweep_{pot,phys,physprec,prec,range}.csv` (script nối hậu tố vào prefix).

Kết quả: RT-DETR-l đạt nửa recall ở ~29–34 px gốc so với ~52 px của YOLOv8-s.

## M4 / M4b — ngân sách giám sát

```bash
# M4: rút số khung xuống 25% và 50%
"$PY" exp_data_scaling.py --arm nearfar --fracs 0.25 0.5 --seeds 0 1 2 \
  --labels-dir "$LAB" --images-dir "$IMG" --imgsz 1280 --epochs 200 --patience 20 \
  --batch 8 --workers 2 --device 0 --out results/rebuttal/M4_data_scaling.csv

# M4b: giữ nguyên số khung, chỉ đổi thành phần — lặp khung có cây xa 20 lần
"$PY" exp_data_scaling.py --arm nearfar --oversample-far 20 --seeds 0 1 2 \
  --labels-dir "$LAB" --images-dir "$IMG" --imgsz 1280 --epochs 200 --patience 20 \
  --batch 8 --workers 2 --device 0 --out results/rebuttal/M4b_oversample_far.csv
```

→ `M4_data_scaling.csv` + `M4_train_scale.csv`, `M4b_oversample_far.csv`, và với mỗi cái
là `_eval.csv` / `_eval_curve.csv` do `eval_testset.py` sinh ở bước cuối.

⚠️ `--workers 2`: mặc định 8 gây deadlock fork giữa OpenCV và DataLoader trên máy này —
training đứng hẳn sau vài epoch, GPU 0%, không báo lỗi.

## M7 — phổ công suất của hai cảm biến

Đo độ phân giải cảm biến **thực sự giao ra**, độc lập với mọi detector:

```bash
# hai lan chay, khac nhau o --mode
# --mode angular can f/p sau moi duong dan: NHAN=duong/dan=F_P
"$PY" exp_power_spectrum.py --mode angular \
  --dir "1080p=$IMG=1360" --dir "8K=$IMG2=5251" --n-frames 8 --crop 1024 \
  --out results/rebuttal/M7_spectrum_angular.csv

"$PY" exp_power_spectrum.py --mode nyquist \
  --dir "1080p=$IMG" --dir "8K=$IMG2" --n-frames 8 --crop 1024 \
  --out results/rebuttal/M7_spectrum_nyquist.csv

"$PY" exp_power_spectrum.py --selftest   # kiem tren anh tong hop truoc khi tin so that
```

So ở **tần số góc** cố định (510 và 170 chu kỳ/radian) chứ không ở phần trăm Nyquist
riêng từng khung — so theo Nyquist là so hai thang góc khác nhau, vô nghĩa khi đối chiếu
hai camera.

## M9 — chi phí suy luận

```bash
"$PY" exp_compute_cost.py --images-dir "$IMG" --runs runs --tag nearfar --seed 0 \
  --sizes 640 960 1280 1920 2560 3840 --device 0 \
  --out results/rebuttal/M9_compute.csv
```

→ `M9_compute.csv`. Con số tuyệt đối là lớp máy trạm (RTX 5060 Ti), chỉ **tỉ lệ giữa
các dòng** chuyển sang bo mạch nhúng được.

---

## Phép đảo dấu cắt ô ở 5 seed

Ba CSV dưới đây là kết quả phép kiểm, không phải kết quả đo; đầu vào của chúng nằm ở
`results/baselines/sahi_perseed_1080p_n5.csv` và `results/crosssite/*_perseed_n5.csv`
(lệnh sinh ra chúng ghi ở `results/crosssite/README.md`).

```bash
# hai nửa của phép đảo dấu + phép kiểm tương tác
"$PY" exp_seed5_stats.py \
  --p1080 results/baselines/sahi_perseed_1080p_n5.csv \
  --p8k   results/crosssite/hd_sahi_perseed_n5.csv \
  --out   results/rebuttal/tiling_reversal_stats.csv

# đối chứng E5: cùng bộ 8K, khung gốc so với khung đã hạ mẫu
"$PY" exp_seed5_stats.py \
  --p1080 results/crosssite/E5_sahi_8k_ds1080_perseed_n5.csv \
  --p8k   results/crosssite/hd_sahi_perseed_n5.csv \
  --label-1080 "8K ha mau 1920x1080 (tile 320)" \
  --label-8k   "8K goc (tile 1280)" \
  --out   results/rebuttal/E5_control_stats.csv
```

## Nâng mọi nhánh lên 5 seed

```bash
"$PY" train_missing_seeds.py --seeds 0 1 2 3 4 --workers 2 --device 0 \
  --out results/rebuttal/seed_manifest.csv
```

→ `seed_manifest.csv`: mỗi dòng là một lượt train (nhánh, seed, script, số giây, mã trả
về, có `best.pt` hay không). Ghi lại sau **mỗi** lượt chứ không phải cuối cùng, nên mất
điện giữa chừng vẫn còn vết.

## M4 — vì sao "ít dữ liệu tốt hơn" không phải kết quả thật

Ở 5 seed, recall tầng xa của nhánh 25% dữ liệu (0,420) cao hơn nhánh đầy đủ (0,312), và phép
Welch trên năm giá trị theo seed cho p=0,017. Hai phép kiểm dưới đây cho thấy đó là hiện vật.

**Không phải do trần phát hiện.** Nới trần từ 1000 lên 5000 không đổi gì:

```bash
"$PY" eval_testset.py --labels-dir "$LAB" --images-dir "$IMG" --runs runs \
  --tags scale025 scale050 nearfar --seeds 0 1 2 3 4 --iou 0.3 --conf 0.001 \
  --imgsz 1280 --max-det 5000 --device 0 \
  --out results/rebuttal/M4_capcheck_md5000.csv
```

→ `M4_capcheck_md5000.csv` (+ `_curve.csv`). Far recall giống hệt ở cả hai trần:
0,4195 / 0,3902 / 0,3122.

**Là do chọn sai đơn vị phân tích.** Tầng xa chỉ có 82 hộp thuộc 66 cây riêng biệt; năm seed
chấm lại đúng 66 cây đó, nên phương sai giữa seed đo lần chạy chứ không đo mẫu:

```bash
"$PY" exp_cluster_ci.py --labels-dir "$LAB" --images-dir "$IMG" --runs runs \
  --tags scale025 nearfar --seeds 0 1 2 3 4 --iou 0.3 --conf 0.001 --imgsz 1280 \
  --max-det 1000 --device 0 --out results/rebuttal/M4_cluster_ci.csv \
  --plants-out results/dataset/distinct_plants.csv
```

| nhánh | far recall | bootstrap theo cây | theo clip |
|---|---|---|---|
| scale025 | 0,420 | [0,317 – 0,527] | [0,193 – 0,881] |
| nearfar  | 0,312 | [0,209 – 0,429] | [0,076 – 0,793] |

Hai khoảng chồng nhau trên phần lớn độ dài; mỗi khoảng rộng gấp ~2 lần con số 0,107 ngăn cách
hai trung bình. Bài vì thế báo cáo **không có hiệu ứng** của kích thước tập huấn luyện, chứ
không phải hiệu ứng nghịch.
