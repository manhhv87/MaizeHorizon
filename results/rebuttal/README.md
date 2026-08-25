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

## Khoảng tin cậy cho h50

`h50` là đại lượng trung tâm của bài nhưng trước giờ chỉ báo một con số điểm, nên không trả
lời được câu hỏi hiển nhiên nhất: chênh 1,7 px giữa input 960 và 1920 có phân biệt được với
0 không.

```bash
"$PY" exp_h50_ci.py --labels-dir "$LAB" --images-dir "$IMG" --runs runs \
  --tag nearfar --seeds 0 1 2 3 4 --imgsz-list 640 960 1280 1920 \
  --iou 0.3 --conf 0.25 --max-det 1000 --boot 2000 --device 0 \
  --out results/scaling/h50_ci.csv
```

| imgsz | h50 | theo cây (95%) | theo clip (95%) |
|---|---|---|---|
| 640 | 59,87 | [57,24 – 62,06] | [55,50 – 64,25] |
| 960 | 52,95 | [51,68 – 54,52] | [52,19 – 54,35] |
| 1280 | 51,90 | [50,59 – 53,35] | [50,80 – 53,04] |
| 1920 | 51,26 | [50,00 – 52,63] | [49,02 – 53,19] |

Đơn vị lấy mẫu lại là **cây**, không phải hộp (một cây cho 3,6 hộp ở tầng gần nên coi chúng
độc lập sẽ thu hẹp khoảng giả tạo) và không phải seed (mọi seed chấm lại đúng những cây ấy,
nên phương sai giữa seed đo lần chạy chứ không đo mẫu).

Đọc: 640 và 960 **không giao nhau** → bước đó là thật.

⚠️ **Cách đọc cũ ở đây là sai, đã thay bằng M8.** Trước đây mục này kết luận "960 và 1920
chồng nhau gần hết → không phân giải được", rồi bài dùng nó để khẳng định đầu vào không thêm
gì trên 960. Hai khoảng biên chồng nhau **không** chứng minh hai giá trị bằng nhau. Bootstrap
ghép cặp chính hiệu số (M8) cho −1,69 px [−2,48, −0,96] p=0,0005 trên chính nhánh này, và
−5,66 px trên `stock`. Bước 960→1920 là thật; điểm gãy nằm ở **độ phân giải gốc**, không phải
ở 960. Xem mục M8 bên dưới.

---

## M8 — Điểm gãy của quy luật min nằm ở độ phân giải gốc

`exp_h50_contrast.py` → `M8_h50_*.csv`, gộp ở `M8_h50_crossover.csv`

Hai khoảng tin cậy chồng nhau **không** chứng minh hai giá trị bằng nhau — đó là lỗi đọc
khoảng phổ biến nhất, và bảng `h50_ci.csv` ở trên rơi đúng vào nó. Phép đúng là bootstrap
chính **hiệu số** trên cùng một lần lấy mẫu lại: cây nào vào mẫu thì vào cho cả hai vế, nên
phương sai chung do mẫu bị khử và khoảng hẹp hơn hẳn khoảng của từng vế.

Chạy ở **hai ngưỡng**: `conf=0.25` là điểm vận hành của bài, nhưng `h50` ở đó trộn lẫn khả
năng phát hiện với hiệu chuẩn điểm số. `conf=0.001` là trần, gần như mọi dự đoán đều qua
ngưỡng, nên đo khả năng phát hiện gần như thuần tuý. Kết luận chỉ chắc khi hiệu số còn ở cả hai.

```bash
C="--labels-dir $LAB --images-dir $IMG --runs runs --seeds 0 1 2 3 4 --device 0 --boot 2000"

# Dưới mức gốc: thêm điểm ảnh đầu vào có dời sàn không
"$PY" exp_h50_contrast.py $C --tag-a stock   --tag-b stock   --imgsz-a 1920 --imgsz-b 960 \
  --out results/rebuttal/M8_h50_stock_1920v960.csv
"$PY" exp_h50_contrast.py $C --tag-a nearfar --tag-b nearfar --imgsz-a 1920 --imgsz-b 960 \
  --out results/rebuttal/M8_h50_nearfar_1920v960.csv

# Trên mức gốc: nếu cảm biến ấn định sàn thì h50 phải phẳng sau 1920
"$PY" exp_h50_contrast.py $C --tag-a stock   --tag-b stock   --imgsz-a 2560 --imgsz-b 1920 \
  --out results/rebuttal/M8_h50_stock_2560v1920.csv
"$PY" exp_h50_contrast.py $C --tag-a stock   --tag-b stock   --imgsz-a 3840 --imgsz-b 1920 \
  --out results/rebuttal/M8_h50_stock_3840v1920.csv
"$PY" exp_h50_contrast.py $C --tag-a nwd     --tag-b nwd     --imgsz-a 2560 --imgsz-b 1920 \
  --out results/rebuttal/M8_h50_nwd_2560v1920.csv
"$PY" exp_h50_contrast.py $C --tag-a nearfar --tag-b nearfar --imgsz-a 2560 --imgsz-b 1920 \
  --out results/rebuttal/M8_h50_nearfar_2560v1920.csv

# Đổi hàm mất mát có dời sàn không (nwd và stock cùng pool dữ liệu)
"$PY" exp_h50_contrast.py $C --tag-a nwd --tag-b stock --imgsz 1280 \
  --out results/rebuttal/M8_h50_nwd_vs_stock.csv
```

Quét đầu vào **vượt** mức gốc, chưa từng chạy trước đây:

```bash
"$PY" exp_resolution_sweep.py --labels-dir "$LAB" --images-dir "$IMG" --runs runs \
  --tag stock --seeds 0 1 2 3 4 --imgsz-list 1920 2560 3840 --iou 0.3 --conf 0.25 \
  --h50-min-n 100 --device 0 --out-prefix results/rebuttal/M8_stock_above
```

Và quét đầy đủ cho hai nhánh chưa có, để `nearfar` không còn là nhánh duy nhất chống đỡ C1:

```bash
"$PY" exp_resolution_sweep.py --labels-dir "$LAB" --images-dir "$IMG" --runs runs \
  --tag stock --seeds 0 1 2 3 4 --imgsz-list 640 960 1280 1920 --iou 0.3 --conf 0.25 \
  --h50-min-n 100 --device 0 --out-prefix results/rebuttal/M8_stock_sweep
"$PY" exp_h50_ci.py --labels-dir "$LAB" --images-dir "$IMG" --runs runs \
  --tag stock --seeds 0 1 2 3 4 --imgsz-list 1280 --iou 0.3 --conf 0.25 --max-det 1000 \
  --boot 2000 --device 0 --out results/rebuttal/M8_stock_h50_ci.csv

"$PY" exp_resolution_sweep.py --labels-dir "$LAB" --images-dir "$IMG" --runs runs \
  --tag nwd --seeds 0 1 2 3 4 --imgsz-list 640 960 1280 1920 --iou 0.3 --conf 0.25 \
  --h50-min-n 100 --device 0 --out-prefix results/rebuttal/M8_nwd_sweep
"$PY" exp_h50_ci.py --labels-dir "$LAB" --images-dir "$IMG" --runs runs \
  --tag nwd --seeds 0 1 2 3 4 --imgsz-list 1280 --iou 0.3 --conf 0.25 --max-det 1000 \
  --boot 2000 --device 0 --out results/rebuttal/M8_nwd_h50_ci.csv
```

Gộp bảy phép tương phản trên thành một bảng để bài trích dẫn một chỗ:

```bash
"$PY" collect_h50_crossover.py --out results/rebuttal/M8_h50_crossover.csv
```

### `h50` (px gốc) theo đầu vào, ba nhánh, conf 0,25

| imgsz | stock | nwd | nearfar |
|---|---|---|---|
| 640 | 56,64 | 52,69 | 59,87 |
| 960 | 49,75 | 45,88 | 52,95 |
| 1280 | 47,11 | 42,13 | 51,90 |
| **1920 (gốc)** | **44,08** | **39,18** | **51,26** |
| 2560 | 49,74 | 42,57 | 51,86 |
| 3840 | 68,29 | — | — |

### Đọc

**Dưới mức gốc, đầu vào dời sàn thật.** Ghép cặp ở conf 0,25: `stock` 960→1920 cho
−5,66 px [−6,97, −4,61] p=0,0005; `nearfar` cho −1,69 px [−2,48, −0,96] p=0,0005. Cả hai
đều khác 0. Câu "đầu vào ≥960 không thêm gì" trong bản thảo là **sai** — nó rút ra từ chỗ
hai khoảng biên chồng nhau, mà chồng nhau thì không kết luận được gì.

**Hiệu ứng phụ thuộc nhánh, và bài đã chọn nhánh yếu nhất.** Mọi `scaling_*.csv` chạy
`--tag nearfar`, nhánh cho hiệu ứng nhỏ nhất trong ba. Trên `stock` hiệu ứng lớn gấp 3,3 lần.
Bình nguyên là tính chất của +Mint, không phải của cảm biến 1080p.

**Trên mức gốc, đầu vào hết tác dụng.** Tại trần, không nhánh nào cải thiện: `stock` +0,81
[−0,04, +1,56] ns; `nwd` +0,06 [−0,22, +0,42] ns; `nearfar` +2,79 [+0,40, +8,23] **xấu đi**.
Ở điểm vận hành còn hỏng nặng hơn (`stock` h50 44 → 68 px ở 3840) vì lệch thang train–test.
Đây là quy luật min ở dạng sắc nhất bài có được: điểm gãy nằm ở **độ phân giải gốc**, một mốc
vật lý, chứ không phải con số 960. Trên 8K, đầu vào 2560 vẫn còn xa mức gốc 7680 nên recall
còn tăng — hoàn toàn nhất quán.

**NWD dời sàn thật.** `nwd` và `stock` dùng chung `pool_stock_split` (docstring
`train_nwd.py` ghi `nearfar` nhưng lệnh đã chạy dùng stock), nên chỉ hàm mất mát khác nhau.
Hiệu số −4,98 px [−6,18, −4,01] p=0,0005 ở conf 0,25, còn −2,36 px [−3,54, −0,20] p=0,040 ở
trần. Còn ở cả hai ngưỡng → không phải hiệu chuẩn. Việc này không phá C1 (sàn vẫn bị cảm biến
chặn, NWD chỉ trích được nhiều hơn từ cùng số điểm ảnh) nhưng NWD không còn là kết quả âm.

## `*_stats_fpfilt.csv` — phép kiểm cho phép đảo dấu, sau khi chuyển sang COCO

Tính lại từ các CSV từng seed đã chạy với `--fp-size-filter`.

```bash
"$PY" exp_seed5_stats.py \
  --p1080 results/baselines/sahi_perseed_1080p_n5_fpfilt.csv \
  --p8k   results/crosssite/hd_sahi_perseed_n5_fpfilt.csv \
  --tier far --out results/rebuttal/tiling_reversal_stats_fpfilt.csv

"$PY" exp_seed5_stats.py \
  --p1080 results/crosssite/E5_sahi_8k_ds1080_perseed_n5_fpfilt.csv \
  --p8k   results/crosssite/hd_sahi_perseed_n5_fpfilt.csv \
  --tier far --label-1080 "8K ha mau (tile 320)" --label-8k "8K goc (tile 1280)" \
  --out results/rebuttal/E5_control_stats_fpfilt.csv
```

| | Δ AP tầng xa | t | p |
|---|---|---|---|
| 1080p (cảm biến ràng buộc) | −0,207 | −10,16 | 0,001 |
| 8K (đầu vào ràng buộc) | +0,285 | +17,50 | <0,001 |
| **tương tác** | **+0,492** | **18,87** | **<0,001** |
| đối chứng E5 (cùng cảnh, cùng thang ô) | **+0,361** | **16,26** | **<0,001** |

Cả hai nửa nay đều dứt khoát, kể cả nửa 1080p vốn nằm sát ngưỡng ở bản cũ. Hiệu ứng lớn
hơn bản cũ khoảng 15 lần về trị tuyệt đối.
