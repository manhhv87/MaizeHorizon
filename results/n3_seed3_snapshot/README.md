# Bản chụp 3-seed — KHÔNG xoá

33 CSV ở đây là **bản 3 seed** của mọi kết quả phụ thuộc seed, chụp ngày **2026-08-22**
ngay trước khi chạy lại toàn bộ ở 5 seed.

## Vì sao phải giữ

Lượt `rerun_after_relabel.py --seeds 0 1 2 3 4` ghi **đè lên đúng các đường dẫn cũ**
(`results/detection/testset_ap03.csv`, `results/scaling/scaling_*.csv`, …). Nếu không chụp
lại thì mọi con số đang nằm trong bản thảo hiện tại mất chỗ truy vết ngay khi lượt đó chạy,
và trong khoảng thời gian giữa lúc CSV đổi và lúc bản thảo được cập nhật thì không ai đối
chiếu được bài với dữ liệu.

Ngoài ra đây là câu trả lời sẵn cho một câu hỏi phản biện gần như chắc chắn sẽ có:
*thêm seed thì con số đổi thế nào?* — đối chiếu thư mục này với `results/` là ra.

## Cách đọc

Cấu trúc phản chiếu `results/`, bỏ một cấp:

    results/n3_seed3_snapshot/detection/testset_ap03.csv   <->  results/detection/testset_ap03.csv
    results/n3_seed3_snapshot/scaling/scaling_phys.csv     <->  results/scaling/scaling_phys.csv

Mọi file ở đây đều là **seed 0, 1, 2**. Cột `n_seeds` (nếu có) bằng 3.

## Lệnh đã sinh ra chúng

Đúng các lệnh trong `rerun_after_relabel.py` nhưng ở `SEEDS = [0, 1, 2]` (giá trị mặc định
trước ngày 2026-08-22):

```bash
"$PY" rerun_after_relabel.py            # mac dinh --seeds 0 1 2
```

Bản 5 seed thay thế chúng:

```bash
"$PY" train_missing_seeds.py --seeds 0 1 2 3 4     # train seed con thieu cho MOI nhanh
"$PY" rerun_after_relabel.py --seeds 0 1 2 3 4     # ghi de results/ bang ban 5 seed
```

## Lý do chuyển sang 5 seed

Ở 3 seed, phép đảo dấu cắt ô không nửa nào đạt ý nghĩa thống kê (p=0,158 và p=0,103), và
đối chứng sạch E5 cũng không (p=0,13). Ở 5 seed cả ba đều đạt (p=0,050; p=0,002; p=0,0002).
Seed thêm được áp cho **mọi** nhánh chứ không riêng nhánh nào, để không đọc được thành
dừng lại khi kết quả trở nên thuận lợi — xem `results/rebuttal/seed_manifest.csv` để biết
đã train gì và mất bao lâu.
