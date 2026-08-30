# `results/detection/` — bảng phát hiện theo tầng

```bash
PY=/home/manhhv/anaconda3/envs/MAAF-NET/bin/python
LAB="data/test/labels";  IMG="data/test/images"
ARMS="stock nearonly nearfar distill distill_shuffle distill_synth nwd"
```

Toàn bộ nhóm này sinh bởi `rerun_after_relabel.py`, là nguồn sự thật cho biết CSV nào
ứng với lệnh nào. Chạy lại tất cả:

```bash
"$PY" rerun_after_relabel.py --seeds 0 1 2 3 4     # bo qua --seeds de dung mac dinh 0 1 2
"$PY" rerun_after_relabel.py --dry                 # chi in lenh, khong chay
```

Từng lệnh một, nếu chỉ cần một bảng:

```bash
# tab:detection, tab:tier — recall/precision tai diem van hanh conf 0.25
"$PY" eval_testset.py --labels-dir "$LAB" --images-dir "$IMG" --runs runs \
  --tags $ARMS --seeds 0 1 2 3 4 --iou 0.3 --conf 0.25 --imgsz 1280 --device 0 \
  --out results/detection/testset_iou03.csv          # va _iou05 voi --iou 0.5

# tab:cliff — tran recall, conf 0.001
"$PY" eval_testset.py --labels-dir "$LAB" --images-dir "$IMG" --runs runs \
  --tags $ARMS --seeds 0 1 2 3 4 --iou 0.3 --conf 0.001 --imgsz 1280 --device 0 \
  --out results/detection/testset_ceiling.csv

# AP theo tang
"$PY" eval_ap.py --labels-dir "$LAB" --images-dir "$IMG" --runs runs \
  --tags $ARMS --seeds 0 1 2 3 4 --iou 0.3 --imgsz 1280 --device 0 \
  --out results/detection/testset_ap03.csv           # va _ap05 voi --iou 0.5
```

`*_curve.csv` do chính `eval_testset.py` sinh kèm, không phải một lệnh riêng.

## `testset_ceiling_md1000.csv` — kiểm tra trần phát hiện

Ở `conf=0.001`, trần `max_det=300` chạm 100% số khung, vì nó áp lên tổng box **cả hai
lớp** mà lớp `ignore` chiếm 45–58% ngân sách rồi bị evaluator vứt đi. Nới trần lên 1000
để đo xem con số trong bài có bị trần ràng buộc không:

```bash
"$PY" eval_testset.py --labels-dir "$LAB" --images-dir "$IMG" --runs runs \
  --tags $ARMS --seeds 0 1 2 3 4 --iou 0.3 --conf 0.001 --imgsz 1280 \
  --max-det 1000 --device 0 --out results/detection/testset_ceiling_md1000.csv
```

Kết quả: trần chỉ ràng buộc `stock` (far ceiling 0,337→0,415 nhưng far precision sụp
0,515→0,287); `nearfar` +0,004; `distill`/`distill_shuffle` đúng 0,0000.

## `cluster_ci.csv` — khoảng tin cậy có tính đến gộp cụm

Các box không độc lập: khung cách nhau 1/5 giây nên một cây vật lý cho nhiều box.

```bash
"$PY" exp_cluster_ci.py --labels-dir "$LAB" --images-dir "$IMG" --runs runs \
  --tags stock nearfar --seeds 0 1 2 3 4 --iou 0.3 --conf 0.001 --imgsz 1280 \
  --max-det 1000 --device 0 --out results/detection/cluster_ci.csv \
  --plants-out results/dataset/distinct_plants.csv
```

Sinh kèm `results/dataset/distinct_plants.csv` (xem README ở đó).

## `pr_far_curve.csv` — đường cong PR tầng xa (dữ liệu của Hình 3)

`plot_pr_far.py --curve-out` ghi đường cong đã nội suy lên lưới 201 điểm recall.

Trước đây script chỉ vẽ hình, không ghi file nào, nên mọi khẳng định trong chú thích Hình 3
không kiểm lại được bằng gì ngoài việc **nhìn hình**. Và nhìn hình đã sai: chú thích viết
"+Mint và +Distill nằm dưới Stock trên suốt đường cong", nhưng đối chiếu số cho thấy +Mint
**vượt lên** ở 11 điểm, trong khoảng recall 0,18–0,27, nhiều nhất +0,04 precision.

```bash
"$PY" plot_pr_far.py --labels-dir "$LAB" --images-dir "$IMG" --runs runs \
  --tags stock nwd nearfar distill --seeds 0 1 2 3 4 --iou 0.3 --device 0 \
  --out paper/figures/fig_pr_far --curve-out results/detection/pr_far_curve.csv
```

Script tự đối chiếu và in ra khi chạy:

| Nhánh so với Stock | Điểm trên | Điểm dưới | Kết luận |
|---|---|---|---|
| `nwd` | **87** | **0** | trội trên toàn bộ |
| `nearfar` | 11 | 62 | **cắt nhau** |
| `distill` | 0 | 81 | nằm dưới |

⚠️ Hình này dùng trần **1000** phát hiện mỗi khung (mặc định của `plot_pr_far.py`), khác Hình
cliff dùng **300**. Cả hai chú thích nay đều nêu rõ trần của mình.
