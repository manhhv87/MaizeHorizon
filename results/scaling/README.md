
## Quét sáu điểm, gồm hai mức trên độ phân giải gốc

`scaling_*.csv` nay chạy tới **3840**, không dừng ở 1920 như trước.

Bốn dòng cũ (640/960/1280/1920) **giữ nguyên tuyệt đối** — chỉ thêm hai kích thước. Kiểm bằng
cách so từng ô trước khi thay.

```bash
"$PY" exp_resolution_sweep.py --labels-dir "$LAB" --images-dir "$IMG" --runs runs \
  --tag nearfar --seeds 0 1 2 3 4 --imgsz-list 640 960 1280 1920 2560 3840 \
  --iou 0.3 --conf 0.25 --device 0 --h50-min-n 100 \
  --out-prefix results/scaling/scaling

"$PY" plot_scaling.py --prefix results/scaling/scaling --out paper/figures/fig_scaling
"$PY" plot_scaling.py --prefix results/scaling/scaling --lang vi --out paper/vi/figures/fig_scaling
```

### `h50` theo đầu vào — hình chữ U, đáy tại mức gốc

| imgsz | 640 | 960 | 1280 | **1920** | 2560 | 3840 |
|---|---|---|---|---|---|---|
| `h50` (px gốc) | 59,87 | 52,95 | 51,90 | **51,26** | 51,86 | 52,56 |

Đầu vào mua được khả năng phát hiện chừng nào nó còn khôi phục điểm ảnh cảm biến đã thu, chạm
đáy đúng tại mức gốc **1920**, rồi **quay đầu**. Đây là bằng chứng trực quan cho quy luật min.

⚠️ Hiệu ứng trên mức gốc chỉ khoảng **1,3 px** nên **vô hình** ở panel A và B (trục rộng 160 px).
Vì vậy hình có thêm **panel C** vẽ thẳng `h50` theo đầu vào — đó là chỗ duy nhất nhìn ra được
điểm gãy. Thứ nhìn thấy ở panel B là đường 3840 tụt trên cây đã phân giải (>100 px), tức lệch
thang train–test, không phải sàn thay đổi.
