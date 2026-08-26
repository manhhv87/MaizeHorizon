# Cross-site: Đan Phượng (Hà Nội) → Nam Sách (Hải Phòng)

Kiểm chứng trên **địa điểm và cảm biến độc lập**, chạy 2026-08-16 sau khi
Biosystems Engineering desk-reject (Ms. YBENG-D-26-01644) với lý do nêu trong
editorial của chính ban biên tập:

> *"Research that is not validated on independent data is a proof of concept.
> It is not engineering... Singular datasets, even with some form of
> cross-validation, are rarely considered to provide proof of generalisation."*
> — Demeyer et al., Biosystems Engineering 269 (2026) 104524

## Hai bộ dữ liệu

| | Đan Phượng (bài hiện tại) | Hải Phòng (mới) |
|---|---|---|
| **Địa điểm** | Đan Phượng, Hà Nội | **Nam Sách, Hải Phòng** (~70 km về phía đông) |
| **Thời gian quay** | **4/2025** | **7/2026** |
| Khung tách | 2026-06-28 | — |
| Cảm biến | Logitech C920, 1920×1080 | Samsung S23, 7680×4320 (8K) |
| Khung có nhãn | 120 | 150 (2 phiên) |
| Box plant | 4.067 | 29.297 |
| Box ignore | 941 | 6.482 |
| Tầng far @1280 | 82 | 5.790 |
| Dưới 12px POT | **4** | **2.324** |

⚠️ **Bản thảo hiện KHÔNG nêu năm quay ở bất kỳ đâu** — chỉ "over three days at
several times of day". Hướng dẫn Vallejo đòi mô tả dữ liệu phải có "environmental/
contextual conditions", mà với dữ liệu nông nghiệp thì thời điểm là cốt lõi. Phải bổ sung.

Với hai năm + hai tỉnh + hai cảm biến, bộ dữ liệu đáp ứng **cả hai vế** của tiêu chí
Demeyer (*"multiple years **and/or** multiple sites"*), không còn phải dựa vào chữ *or*.

Nhãn Hải Phòng do **người gán**, cùng lược đồ lớp (`0: plant`, `1: ignore`) và
cùng giao thức ignore. Nguồn: `/media/manhhv/DATA/AI/_archive/paper2/test2/`.

## Lệnh đã chạy

```bash
LAB2=/media/manhhv/DATA/AI/_archive/paper2/test2/labels
IMG2=/media/manhhv/DATA/AI/_archive/paper2/test2/images

# E1 — chuyển vùng zero-shot, 8K gốc
"$PY" eval_testset.py --labels-dir "$LAB2" --images-dir "$IMG2" \
  --runs runs --tags stock --seeds 0 1 2 --iou 0.3 --conf 0.001 \
  --imgsz 1280 --max-det 3000 --device 0 \
  --out results/crosssite/namsach_ceiling.csv

# Đối chứng cảm biến — cùng cảnh, hạ mẫu 8K -> 1920x1080 (LANCZOS), rồi eval y hệt
#   script hạ mẫu: xem mục "Tái lập" bên dưới
"$PY" eval_testset.py --labels-dir <hd_1080>/labels --images-dir <hd_1080>/images \
  --runs runs --tags stock --seeds 0 1 2 --iou 0.3 --conf 0.001 \
  --imgsz 1280 --max-det 3000 --device 0 \
  --out results/crosssite/namsach_ds1080_ceiling.csv
```

⚠️ `--max-det 3000`, **không** phải 300 hay 1000. Hải Phòng có ~238 box/khung so
với ~34 của bộ cũ; trần mặc định cắt cụt nghiêm trọng. Đây là biến thể của bẫy `max_det`
đã biết của repo.

## Kết quả — recall theo tầng (conf=0.001, IoU 0.3, imgsz 1280, 3 seed)

| Tầng | Đan Phượng | Hải Phòng 8K | n_gt (HD) |
|---|---|---|---|
| near | 0.981 | 0.916 ± 0.034 | 11.775 |
| mid | 0.771 | 0.678 ± 0.084 | 11.732 |
| far | 0.337 | 0.183 ± 0.095 | 5.790 |

## Kết quả — đường cong recall theo POT

| POT | Đan Phượng 1080p | Hải Phòng 8K | HD hạ mẫu 1080p |
|---|---|---|---|
| 6 | — | 0.017 (n=231) | 0.009 |
| 10 | 0.000 (n=4) | 0.088 (n=2.092) | 0.043 |
| 14 | 0.355 (n=78) | 0.251 (n=3.466) | 0.154 |
| 18 | 0.476 (n=164) | 0.475 (n=3.546) | 0.367 |
| 22 | 0.618 (n=303) | 0.683 (n=3.204) | 0.592 |
| 28 | 0.894 (n=771) | 0.819 (n=4.982) | 0.757 |
| 56 | 0.990 (n=661) | 0.934 (n=2.837) | 0.927 |

**h50 (POT @imgsz 1280, conf=0.001): 18.7 · 18.5 · 20.4**

## Ba kết luận

**1. Định luật tái lập qua hai tỉnh và hai cảm biến.** h50 lệch 18.7 vs 18.5 —
1%. Tại vùng chuyển tiếp (POT 18) hai đường trùng: 0.476 vs 0.475.

**2. Khẳng định sub-12px giờ có sức mạnh thống kê.** Ô 8–12 POT: từ n=4
(recall 0.000, vô nghĩa) lên n=2.092 (recall 0.088). Ô 4–8: n=231, recall 0.017.
Phát biểu đúng không phải "về 0" mà là "~9% rồi ~2%".

**3. Hiệu ứng độ phân giải cảm biến là thật nhưng nhỏ, và chỉ ở tầng nhỏ.**
Cùng cảnh, cùng cây, cùng ánh sáng, chỉ khác độ phân giải: +0.046 (8–12 POT),
+0.097 (12–16), +0.108 (16–20), +0.022 (32–48), −0.007 (64–96). h50 dịch
18.5 → 20.4, tức **10%**, dù số pixel danh nghĩa gấp **4 lần**.

## Cảnh báo diễn giải — đọc trước khi trích số

**a) `h50` khớp 18.7 vs 18.5 một phần là trùng hợp.** Tách ra thì:
ruộng Hải Phòng **khó hơn ~9%** ở cùng độ phân giải (20.4 vs 18.7), rồi cảm biến
8K **bù lại ~10%** (20.4 → 18.5). Hai hiệu ứng ngược chiều xấp xỉ triệt tiêu.
Viết vào bài phải viết như vậy — tách được hai hiệu ứng có giá trị hơn con số
trùng khớp.

**b) Phải so cùng `conf`.** `h50 ≈ 52px native` trong bài đo ở `conf=0.25`
(≈ 34.7 POT @1280). Mọi số ở đây đo ở `conf=0.001`. So chéo hai ngưỡng sinh ra
một "hiệu ứng địa điểm ~70%" hoàn toàn giả — đã mắc lỗi này một lần khi phân tích.

**c) 8K của điện thoại phần lớn là phóng đại rỗng.** Phổ công suất theo bán kính
(crop 1024², 8 khung mỗi bộ): tại 0.75 Nyquist, S23 8K giữ tỷ lệ công suất
**0.0025**, C920 1080p giữ **0.0345** — gấp 14 lần. Đây là lý do 4× pixel chỉ
mua được 10% h50, và là **trụ cột lập luận**, không phải phần trang trí: thiếu
nó, dữ liệu mới trông như đang phản bác luận điểm của bài.

**d) Chiều cao box native không so sánh được giữa hai cảm biến.** 24px trên
1080p và 24px trên 8K là hai góc nhìn khác nhau. Chỉ POT (và góc nhìn, nếu biết
`f/p`) mới so được. `f/p` của S23 khi quay 8K **chưa đo** — FOV lúc quay 8K hẹp
hơn lúc chụp ảnh, không tra được từ thông số.

## Tái lập

Script hạ mẫu (`PIL.Image.LANCZOS`, quality 92, nhãn chuẩn hoá nên giữ nguyên):
xem `scripts/downsample_hd.py`. Đây là **ablation cảm biến**, không phải suy
giảm tổng hợp thay cho ảnh xa thật — ảnh 8K là thật, việc hạ mẫu chỉ để tách
biến độ phân giải khỏi biến địa điểm. Phải khai rõ như vậy trong bài.

---

# E2 — Quét input size trên khung 8K

```bash
"$PY" exp_resolution_sweep.py --labels-dir "$LAB2" --images-dir "$IMG2" \
  --runs runs --tag stock --seeds 0 1 2 --imgsz-list 1280 1920 2560 3840 \
  --iou 0.3 --conf 0.001 --max-det 3000 --device 0 \
  --out-prefix results/crosssite/hd_sweep
```

⚠️ **Chỉ đọc `hd_sweep_pot.csv`.** Hai cột khác không dùng được cho bộ này:
- `hd_sweep_phys.csv` / cột `h50_phys_px`: `PHYS_EDGES` hiệu chuẩn cho ảnh 1920,
  trên 8K mọi box dồn vào bin trên cùng (nguồn cảnh báo `Mean of empty slice`).
  Tính lại h50 từ đường cong POT, chỉ dùng bin có `n_gt >= 100`.
- `recall_near/mid/far` trong `_range.csv`: **bẫy đã biết của repo** — tầng
  định nghĩa theo POT mà POT phụ thuộc imgsz, nên mỗi imgsz đo trên quần thể cây
  khác nhau. Tầng far ở imgsz 3840 chỉ có **12 cây**.

## Recall tại cùng chiều cao native (cách trình bày đúng)

Cùng một cây, cùng một ảnh 8K, chỉ đổi input size:

| h native | 1280 | 1920 | 2560 | 3840 | lãi |
|---|---|---|---|---|---|
| 60px | 0.088 | 0.324 | 0.467 | 0.496 | +0.408 |
| 80px | 0.224 | 0.475 | 0.601 | 0.658 | +0.434 |
| 100px | 0.400 | 0.629 | 0.719 | 0.766 | +0.366 |
| 120px | 0.579 | 0.753 | 0.833 | 0.843 | +0.264 |
| 160px | 0.789 | 0.885 | 0.907 | 0.903 | +0.114 |
| 300px | 0.924 | 0.922 | — | — | ~0 |

**h50 (px native trên cảm biến 8K):** 110.9 · 83.6 · 65.2 · 60.5 → cải thiện
**1.83×** từ imgsz 1280 lên 3840, và **bão hoà**: bước 2560→3840 chỉ được 7%.

## Kết luận — định luật thống nhất

Ràng buộc là **`min(độ phân giải hữu hiệu của cảm biến, input size của mạng)`**.

| Giàn máy | Danh nghĩa | Bão hoà tại | Độ phân giải hữu hiệu |
|---|---|---|---|
| C920, Đan Phượng | 1920 | imgsz ~960 | ≲ 960–1280 |
| S23 8K, Hải Phòng | 7680 | imgsz ~2560–3840 | ~2560–3840 |

Bài hiện tại đo trúng vế **cảm biến chặn**; dữ liệu mới lộ vế **mạng chặn**.
Cả hai là một định luật. Điểm bão hoà của h50 chính là **phép đo độ phân giải
hữu hiệu** của hệ ảnh — độc lập với, và trùng kết luận với, phép đo phổ công suất.

⚠️ **Tiêu đề bài hiện tại sai trong trường hợp tổng quát.** *"Network Input Size
Does Not Lower the Floor"* chỉ đúng khi cảm biến đã cạn thông tin. C1 phải viết
lại thành dạng `min(...)`.

## Việc tiếp theo (đã làm, xem các mục bên dưới)

- **E4 — SAHI trên 8K.** Bài báo cáo tiling *làm giảm* far AP trên 1080p
  (0.020 → 0.003) vì cảm biến không ghi được pixel nào để khôi phục. Trên 8K
  pixel có thật, nên tiling đáng lẽ phải thắng. Đây là dự đoán kiểm chứng được.
  ✅ **Đã chạy — dấu bị lật đúng như dự đoán.**
- **E3 — kiểm `d_max` xuyên cảm biến.** Cần `f/p` của S23 ở chế độ quay 8K; phải
  đo thực địa (thước dài đã biết ở khoảng cách đã biết), không tra được từ
  thông số vì FOV chế độ 8K hẹp hơn chế độ ảnh.

---

# E3 — Hiệu chuẩn hình học và tầm làm việc xuyên cảm biến

## `f/p` của Samsung S23 ở chế độ quay 8K

Thông số Samsung: **80°** ở chế độ 8K. Đọc là **góc chéo** thì
`f/p = (√(7680²+4320²)/2)/tan(40°) = 5251 px`, cho FOV ngang 72.4° và dọc 44.7°.

Kiểm tra độc lập: FOV ngang 72.4° trên khung 16:9 quy về máy phim 35mm là tiêu cự
**24.6 mm** — khớp thông số "24 mm equivalent" Samsung công bố. Vậy 80° là góc chéo,
không phải góc ngang.

## Hiệu chuẩn từ dữ liệu — quan hệ đường chân trời

Với vật **thẳng đứng** đứng trên mặt phẳng, quan hệ sau là **chính xác** và
**không phụ thuộc `f/p` lẫn góc nghiêng**:

    h = (P / H_cam) · (y_base − y_horizon)

Hồi quy trung vị theo dải `y_base` (24 dải, mỗi dải ≥80 box, R² = **0.985**):

    h = 0.1499 · (y_base − 222)

→ **chân trời ở hàng y = 222**, **chiều cao cây trung vi P = 0.097 m**.

Đây là lý do phải dùng quan hệ này thay vì fit trực tiếp `(f/p, θ, P)`: bản fit
trực tiếp chạy tới biên vì nó bỏ qua **độ lệch ngang** — ở FOV 72° một cây ở mép
khung xa hơn cây giữa khung 1.24 lần dù cùng hàng ảnh.

## ⚠️ Góc nghiêng ghi chép sai

Ghi chép hiện trường nói *nghiêng 55° so với trục thẳng đứng* (= 35° dưới phương
ngang). Dữ liệu **bác bỏ** con số đó:

| Góc nghiêng giả định | `f/p` suy ra | FOV ngang |
|---|---|---|
| 35° dưới ngang (ghi chép) | 2768 | **108°** — bất khả, ống siêu rộng không quay 8K |
| **20.3° dưới ngang** | **5251** | **72.4°** ✅ khớp thông số |
| 23.0° dưới ngang | 4576 | 80.0° |

Nghiêng ~20° cũng khớp giàn Đan Phượng (9–22°, trung vị 15°). **Dùng 20.3°.**

## Kiểm chứng: khoảng cách thực của cây đã gán nhãn

Với `f/p=5251`, nghiêng 20.3°, `H_cam=0.65 m`: cây đã gán nhãn nằm ở
**1.0 – 8.0 m** (p5–p95), trung vị 3.2 m. Bài báo cáo Đan Phượng 1.1 – 13.8 m.
Ngắn hơn vì cây nhỏ hơn (0.097 m so với ~0.15 m). Hình học nhất quán.

## Ngưỡng phát hiện là ngưỡng GÓC

`d_max = (f/p) · P / h50_native`, cùng `conf=0.001`, imgsz 1280:

| | h50 native | ngưỡng góc | P dùng | d_max |
|---|---|---|---|---|
| Đan Phượng, C920 | 28.1 px | **1.182°** | 0.15 m ⚠️ | 7.27 m |
| Hải Phòng, S23 8K | 111.0 px | **1.211°** | 0.097 m (fit) | 4.59 m |

⚠️ `P = 0.15 m` cho Đan Phượng là **giả định**, lấy trung điểm dải 0.11–0.19 m mà
bài báo cáo cho các phiên test. `d_max` tỉ lệ tuyến tính với `P`, nên con số 7.27 m
mang nguyên sai số đó (±27% nếu P đi hết dải). Ngưỡng **góc** thì không phụ thuộc
`P` — đó là lý do nên so ngưỡng góc chứ đừng so `d_max` giữa hai địa điểm.

Chênh **2.4%** dù hai cảm biến khác nhau 3.86× về độ phân giải góc. Lý do: hai máy
có FOV gần bằng nhau (70.4° và 72.4°) và cùng chạy imgsz 1280, nên **độ phân giải
góc của đầu vào mạng** như nhau — mạng chặn ở cả hai, cảm biến không đóng vai trò.

⚠️ Đây là **phép thử yếu** cho công thức `d_max`: hai giàn tình cờ bị chặn bởi cùng
một thứ, nên việc hai ngưỡng góc khớp nhau chỉ xác nhận **dạng** của mô hình (ngưỡng
là góc, không phải số pixel tuyệt đối). Nó không kiểm `d_max` độc lập được, vì
`d_max` suy ra *từ* `h50` — nói "tại `d_max` thì recall = 0.5" là đúng theo định
nghĩa. Phép thử thật (**E5**): tính khoảng cách mét của từng cây từ hình học đã hiệu
chuẩn, đo recall theo khoảng cách, xem chỗ cắt 50% có rơi đúng chỗ công thức dự đoán.

## Tầm làm việc theo input size (Hải Phòng)

| imgsz | ngưỡng góc | d_max |
|---|---|---|
| 1280 | 1.211° | 4.59 m |
| 1920 | 0.912° | 6.09 m |
| 2560 | 0.710° | **7.82 m** |
| 3840 | 0.659° | 8.43 m |

Cùng đoạn phim, không đổi camera, không train lại: **1280 → 3840 kéo tầm từ 4.6 m
lên 8.4 m (1.84×)**, bão hoà sau 2560. Bước 2560→3840 tốn gấp 2.25× tính toán để
đổi 8% tầm ⇒ **imgsz 2560 là điểm ngọt** trên giàn này.

---

# E4 — SAHI tiling trên 8K: dấu bị lật

```bash
"$PY" exp_sahi_baseline.py --labels-dir "$LAB2" --images-dir "$IMG2" \
  --runs runs --tag stock --seeds 0 1 2 --tile 1280 --imgsz 1280 --overlap 0.2 \
  --iou 0.3 --conf 0.001 --max-det 3000 --device 0 \
  --out results/crosssite/hd_sahi.csv
```

`tile 1280` + `imgsz 1280` = mỗi tile chạy ở **tỉ lệ 1:1 cảm biến**, tức gấp 6 lần
độ phân giải hữu dụng so với full-frame @1280 trên ảnh 7680. Chạy ~28 tile/ảnh,
nghẽn CPU (giải nén 8K), mất ~35 phút.

| Tầng | full-frame @1280 | SAHI | Δ |
|---|---|---|---|
| near (n=11,775) | 0.7623 | 0.6037 | −0.1586 |
| mid (n=11,732) | 0.4088 | 0.3432 | −0.0656 |
| **far (n=5,790)** | 0.0157 | **0.0382** | **+0.0225** |
| all | 0.5564 | 0.5328 | −0.0236 |

## Dấu bị lật so với 1080p

| | far AP, 1080p (bài) | far AP, 8K |
|---|---|---|
| full-frame | 0.020 | 0.0157 |
| SAHI | **0.003** (−85%) | **0.0382** (+143%) |

Tiling chỉ phóng to pixel. Cảm biến chưa ghi được → phóng to vô ích và có hại
(1080p). Cảm biến đã ghi → tiling bóc ra dùng được (8K). Đây là **chân thứ ba**
độc lập cho quy tắc `min(...)`, bên cạnh phổ công suất (E1) và điểm bão hoà
imgsz (E2), và bằng một cơ chế khác hẳn.

## Cái giá, và một lựa chọn tốt hơn

near AP tụt **21%** trên 8K (so với 7% trên 1080p): cây gần cao tới 1256 px bị cắt
qua nhiều tile ở kích thước tile 1280.

E2 cho thấy **tăng imgsz không phải trả giá đó**: recall của cây 300 px native gần
như không đổi qua các imgsz (0.924 → 0.922) trong khi tầng nhỏ cải thiện mạnh.

⚠️ **So sánh chưa trực tiếp.** SAHI ở đây cho tỉ lệ 1:1 cảm biến (6× full-frame
@1280), còn imgsz 3840 chỉ là 2× hạ mẫu; và E2 xuất **recall** chứ không xuất
**AP**. Muốn kết luận chắc "tăng imgsz tốt hơn tiling" thì phải chạy AP ở imgsz
2560/3840 để so cùng đơn vị. Chưa làm.

---

# E4 — Doi chung scale-matching: quet input tren khung DA HA MAU

Cau hoi: gain cua E2 la do cam bien giao them chi tiet, hay chi vi cay dich vao
dai kich thuoc bieu kien ma mang da hoc (moi detector train o input 1280)?

Khung ha mau 1920x1080 khong mang chi tiet nao tren 1080p. Vi nhan chuan hoa nen
moi cay co POT y het o ca hai nhanh tai moi input -> kich thuoc bieu kien duoc
giu co dinh, chi khac luong chi tiet.

```bash
"$PY" scripts/downsample_hd.py --src "$SRC2" --dst "$SRC2_1080"

"$PY" exp_resolution_sweep.py \
  --labels-dir "$SRC2_1080/labels" --images-dir "$SRC2_1080/images" \
  --runs runs --tag stock --seeds 0 1 2 --imgsz-list 1280 1920 2560 3840 \
  --iou 0.3 --conf 0.001 --max-det 3000 --device 0 \
  --out-prefix results/crosssite/hd1080_sweep
```

⚠️ Doc `hd1080_sweep_pot.csv`, KHONG doc cot `h50_phys_px` cua `_range.csv` —
cung cai bay da ghi o E2. h50 tinh lai tu duong cong POT, chi dung bin n_gt >= 100.

## Ket qua — h50 quy ve chieu cao native trong khung 8K goc (px)

| input | 8K goc | ha mau 1080p |
|---|---|---|
| 1280 | 110.9 | 122.2 |
| 1920 |  83.6 |  92.1 |
| 2560 |  65.2 | 103.1 |
| 3840 |  60.5 |  98.3 |

Nhanh ha mau cai thien toi 1920 (do phan giai khung cua chinh no) roi DUNG.
Nhanh goc di tiep toi 2560. Tai input 2560, cung 14px POT: 0.399 vs 0.047.

=> Scale matching mot minh KHONG giai thich duoc gain cua E2. Cach doc
"input binds tren camera 8K" dung vung.

## Bay: khong retrain duoc neu DATASETS_DIR sai (phat hien 20/08/2026)

`data/arms/*/data.yaml` dung `path: data` va `train.txt` chua duong dan TUONG DOI
(`arms/<tag>/images/...`). Ultralytics chi viet lai duong dan bat dau bang `./`; con lai
no giu nguyen va giai theo thu muc lam viec, khong theo vi tri file yaml.

Hau qua: neu `yolo settings datasets_dir` khong tro dung goc repo thi **moi nhanh** deu
scan ra `0 images, N corrupt` va training dung ngay -- ke ca nhanh goc `nearfar`. Da kiem
tren may nay: `datasets_dir=/home/manhhv/AI/datasets` -> `nearfar` bao 3.349 corrupt.

Cac checkpoint trong `runs/` duoc train khi cau hinh nay con dung. Nguoi clone repo se
khong retrain duoc cho toi khi:

    yolo settings datasets_dir="<duong dan tuyet doi toi goc repo>"

`exp_data_scaling.py` khong phu thuoc cai dat nay: no ghi duong dan tuyet doi vao
train.txt/valid.txt cua tap con, nen chay duoc du DATASETS_DIR tro dau.

---

# E5 — Doi chung cat o "sach": 8K goc vs 8K DA HA MAU

E4 so cat o giua HAI camera, nen ngoai cam bien con hai thu khac di kem: kich thuoc
tile (640 vs 1280 px) va tran phat hien (1000 vs 3000). E5 bo ca hai bang cach so
BEN TRONG bo 8K, giua khung goc va ban sao ha mau Lanczos xuong 1920x1080.

Tile duoc chon de PHU CUNG MOT PHAN khung o ca hai nhanh: 1/6 be ngang.
  7680 / 6 = 1280 px  (khung goc)
  1920 / 6 =  320 px  (khung ha mau)
Nho vay moi tile "nhin" cung mot phan canh, va thu duy nhat con khac la luong chi
tiet cam bien giao ra.

```bash
DS=/media/manhhv/DATA/AI/_archive/paper2/test2_1080

# nhanh goc: dung lai chinh lo E4 (hd_sahi.csv == E5_sahi_8k_native.csv)
"$PY" exp_sahi_baseline.py --labels-dir "$LAB2" --images-dir "$IMG2" \
  --runs runs --tag stock --seeds 0 1 2 --tile 1280 --imgsz 1280 --overlap 0.2 \
  --iou 0.3 --conf 0.001 --max-det 3000 --device 0 \
  --out results/crosssite/E5_sahi_8k_native.csv

# nhanh ha mau
"$PY" exp_sahi_baseline.py --labels-dir "$DS/labels" --images-dir "$DS/images" \
  --runs runs --tag stock --seeds 0 1 2 --tile 320 --imgsz 1280 --overlap 0.2 \
  --iou 0.3 --conf 0.001 --max-det 3000 --device 0 \
  --out results/crosssite/E5_sahi_8k_ds1080.csv
```

| Tang xa (n=5.790) | full-frame | SAHI | Δ | tuong doi |
|---|---|---|---|---|
| 8K goc      | 0.0157 | 0.0382 | +0.0225 | **+143%** |
| 8K ha mau   | 0.0053 | 0.0037 | −0.0016 | **−30%** |

Huong song sot qua doi chung. Do lon thi khong: bo chi tiet cam bien cung keo
baseline cua nhanh ha mau xuong 0.005, nen ca hai muc thay doi deu nho.

---

# Lam lai o n=5 (22/08/2026)

`stock` va `nearfar` nay co seed 3, 4 (duong dan trong train.txt phai TUYET DOI,
va `--workers 2` de tranh deadlock DataLoader). Cac lo chay lai:

```bash
# E4 hai nua, kem gia tri TUNG SEED de chay duoc phep kiem
"$PY" exp_sahi_baseline.py --labels-dir "$LAB" --images-dir "$IMG" \
  --runs runs --tag nearfar --seeds 0 1 2 3 4 --tile 640 --overlap 0.2 --imgsz 1280 \
  --iou 0.3 --conf 0.001 --device 0 \
  --out results/baselines/sahi_ap_n5.csv \
  --per-seed-out results/baselines/sahi_perseed_1080p_n5.csv

"$PY" exp_sahi_baseline.py --labels-dir "$LAB2" --images-dir "$IMG2" \
  --runs runs --tag stock --seeds 0 1 2 3 4 --tile 1280 --imgsz 1280 --overlap 0.2 \
  --iou 0.3 --conf 0.001 --max-det 3000 --device 0 \
  --out results/crosssite/hd_sahi_n5.csv \
  --per-seed-out results/crosssite/hd_sahi_perseed_n5.csv

# E5 nhanh ha mau
"$PY" exp_sahi_baseline.py --labels-dir "$DS/labels" --images-dir "$DS/images" \
  --runs runs --tag stock --seeds 0 1 2 3 4 --tile 320 --imgsz 1280 --overlap 0.2 \
  --iou 0.3 --conf 0.001 --max-det 3000 --device 0 \
  --out results/crosssite/E5_sahi_8k_ds1080_n5.csv \
  --per-seed-out results/crosssite/E5_sahi_8k_ds1080_perseed_n5.csv

# h50 tinh lai tu duong cong POT, chi dung bin n_gt >= 100 (KHONG doc h50_phys_px
# cua _range.csv tren bo 8K -- xem canh bao o muc E4)
"$PY" exp_h50_by_arch.py --prefix results/crosssite/hd_sweep --label "8K-n5" \
  --min-n 100 --out results/crosssite/hd_h50_n5.csv

# phep kiem: tung nua + phep kiem tuong tac
"$PY" exp_seed5_stats.py \
  --p1080 results/baselines/sahi_perseed_1080p_n5.csv \
  --p8k   results/crosssite/hd_sahi_perseed_n5.csv \
  --out   results/rebuttal/tiling_reversal_stats.csv

# ngan sach giam sat o n=5 (train scale025/scale050 seed 3,4 roi danh gia lai)
"$PY" exp_data_scaling.py --arm nearfar --fracs 0.25 0.5 --seeds 0 1 2 3 4 \
  --labels-dir "$LAB" --images-dir "$IMG" --imgsz 1280 --epochs 200 --patience 20 \
  --batch 8 --workers 2 --device 0 \
  --out results/rebuttal/M4_data_scaling_n5.csv
```

⚠️ `exp_seed5_stats.py` bao cao ca `paired` lan `welch`. Ban trong bai (n=3) dung
`welch` khong ghep cap; nhung `full` va `sahi` den tu CUNG mot checkpoint nen chung
ghep cap tu nhien, va `paired` moi la phep kiem dung. Script in ca hai de doi chieu.


---

# File da bi thay the boi ban n=5 (2026-08-24)

Ba file duoi day la ban 3 seed, giu lai de doi chieu. Ban dang dung la ban `_n5`:

| ban cu (n=3)                          | ban dang dung (n=5)                       |
|---------------------------------------|-------------------------------------------|
| `hd_sahi.csv`                          | `hd_sahi_n5.csv`                          |
| `E5_sahi_8k_native.csv`                | `hd_sahi_n5.csv` (cung mot lo chay)       |
| `E5_sahi_8k_ds1080.csv`                | `E5_sahi_8k_ds1080_n5.csv`                |

Con so trong ban thao lay tu ban `_n5`. Ban 3 seed cua moi ket qua khac nam o
`results/n3_seed3_snapshot/`.

## Bản chạy lại theo chuẩn COCO (`*_fpfilt.csv`)

`eval_ap.py` nay lọc kích thước cho false positive và tích phân 101 điểm đúng chuẩn COCO
(xem `results/baselines/README.md`), nên nửa 8K của phép đảo dấu và đối chứng E5 chạy lại
với `--fp-size-filter`.

```bash
# Nửa 8K của phép đảo dấu
"$PY" exp_sahi_baseline.py --labels-dir "$LAB2" --images-dir "$IMG2" --runs runs \
  --tag stock --seeds 0 1 2 3 4 --tile 1280 --imgsz 1280 --overlap 0.2 \
  --iou 0.3 --conf 0.001 --max-det 3000 --device 0 --fp-size-filter \
  --out results/crosssite/hd_sahi_n5_fpfilt.csv \
  --per-seed-out results/crosssite/hd_sahi_perseed_n5_fpfilt.csv

# Đối chứng E5, nhánh hạ mẫu (nhánh gốc dùng lại chính lô trên)
DS=/media/manhhv/DATA/AI/_archive/paper2/test2_1080
"$PY" exp_sahi_baseline.py --labels-dir "$DS/labels" --images-dir "$DS/images" --runs runs \
  --tag stock --seeds 0 1 2 3 4 --tile 320 --imgsz 1280 --overlap 0.2 \
  --iou 0.3 --conf 0.001 --max-det 3000 --device 0 --fp-size-filter \
  --out results/crosssite/E5_sahi_8k_ds1080_n5_fpfilt.csv \
  --per-seed-out results/crosssite/E5_sahi_8k_ds1080_perseed_n5_fpfilt.csv
```

Tầng xa 8K: 0,193 → 0,478 (+148%). Cùng chiều bản cũ nhưng lớn hơn nhiều.

### ⚠️ Chạy lại 2026-08-26 — hai lỗi trong `exp_sahi_baseline.py`

Mọi CSV cắt ô ở trên đã được **sinh lại** sau khi sửa hai lỗi; bản cũ nằm ở
`_archive/stale/sahi_fpscale_bug/` kèm giải thích.

1. Hệ số lọc false positive cứng hoá `1920` thay vì lấy từ `max(W, H)` của chính bộ ảnh.
   Trên 8K nó lệch **4 lần**, bỏ qua mọi false positive cao 24–96 px.
2. `max_det` chỉ áp cho từng ô, không cắt sau khi gộp — nhánh cắt ô được ngân sách gấp
   ~28 lần nhánh toàn khung.

Nay `fp_scale_of()` lấy hệ số từ bộ ảnh và **dừng hẳn** nếu ảnh không đồng nhất kích thước;
`--frame-max-det` cắt sau khi gộp ô, mặc định bằng `--max-det`. Ba phép tự kiểm mới
(`--selftest`, 9/9) buộc hệ số lọc phải trùng hệ số phân tầng, kiểm ở **cả hai** cỡ ảnh.

**Kết quả đổi hẳn:** AP tầng xa trên 8K từ `0,193 → 0,478` (+148%) thành `0,180 → 0,163`
(−9%). Phép đảo dấu không tồn tại — cắt ô làm giảm AP tầng xa ở mọi điều kiện đã thử.
Đối chứng E5 co từ +0,361 xuống +0,060 (p=0,014).
