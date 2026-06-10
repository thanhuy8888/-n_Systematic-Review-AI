# Hướng dẫn chạy lại pipeline & cập nhật chart (tối ưu CPU)

Mục tiêu: regenerate lại các biểu đồ trong `experiments/results/` với số liệu
leakage-free để dán vào report — chạy trên **CPU**.

Có 2 script mới, đều tối ưu cho CPU:

| Script | Dùng khi nào | Thời gian (CPU) |
|---|---|---|
| `experiments/baselines/evaluate_screening_clean_cpu.py` | **Khuyên dùng để refresh chart.** Đúng model đang deploy (PubMedBERT embedding + XGBoost hybrid). Có **cache embedding** nên chạy lại lần 2 gần như tức thì. | Lần đầu ~5–15 phút, các lần sau < 1 phút |
| `experiments/baselines/finetune_pubmedbert_cpu.py` | Fine-tune end-to-end PubMedBERT (hướng "future work", số liệu mạnh hơn). | ~30–90 phút tùy cấu hình |

> Cả hai giữ **nguyên phương pháp leakage-free** và **nguyên cấu hình model** của
> bản gốc, nên số in ra chính là số dán vào **Table 5**. Chúng ghi đè đúng các
> file PNG cũ trong `experiments/results/`.

---

## 0. Chuẩn bị (làm 1 lần)

```bat
cd C:\laragon\www\doantotnghiep
.\venv\Scripts\activate
pip install -r requirements.txt
```

Biết số nhân **vật lý** của CPU (KHÔNG phải số luồng). Ví dụ CPU 4 nhân 8 luồng →
dùng `--threads 4`. Trên Windows xem ở Task Manager → Performance → CPU → "Cores".

---

## 1. Refresh chart nhanh (khuyên dùng)

**Lần đầu** (mã hoá embedding rồi cache lại):
```bat
python experiments\baselines\evaluate_screening_clean_cpu.py --threads 4
```

**Các lần sau** (đổi style chart, tinh chỉnh ngưỡng… dùng lại embedding đã cache,
gần như tức thì):
```bat
python experiments\baselines\evaluate_screening_clean_cpu.py --fast
```

Script sẽ:
- Ghi đè 5 biểu đồ: `transformer_confusion_matrix.png`, `transformer_roc_curve.png`,
  `transformer_pr_curve.png`, `transformer_probability_distribution.png`,
  `model_comparison_chart.png`.
- In ra khối **"COPY THESE INTO TABLE 5"** (Precision/Recall/F1/ROC-AUC/PR-AUC).

### Vì sao nó nhanh trên CPU
1. **Cache embedding** (lớn nhất): PubMedBERT chỉ encode 1 lần, lưu vào
   `experiments/cache/`. Mọi lần chạy sau load từ đĩa < 1 giây.
2. **Pin số thread** = số nhân vật lý (`--threads`), tránh tranh chấp.
3. **Batch encode lớn hơn** + `n_jobs=-1` cho RandomForest, `tree_method="hist"` cho XGBoost.
4. **`--fast`**: bỏ SVC (chậm nhất) và giảm số cây — chỉ dùng khi nghịch style chart,
   **không** dùng khi chốt số cuối (để số khớp đúng cấu hình thesis).

---

## 2. Fine-tune PubMedBERT trên CPU (nếu muốn số mạnh hơn)

Công thức cân bằng tốc độ / độ chính xác (khuyên dùng cho CPU):
```bat
python experiments\baselines\finetune_pubmedbert_cpu.py ^
    --epochs 3 --batch-size 8 --grad-accum 4 ^
    --max-len 256 --freeze-layers 8 --threads 4
```

Công thức **nhanh nhất** (model distil, refresh chart cho lẹ):
```bat
python experiments\baselines\finetune_pubmedbert_cpu.py ^
    --model nlpie/distil-biobert --epochs 2 --batch-size 16 ^
    --max-len 224 --freeze-layers 3 --threads 4
```

Ghi đè 4 chart transformer + lưu model vào `sr_core/screening_model/finetuned_pubmedbert/`
+ in khối Table 5.

### Các cờ tối ưu CPU (ý nghĩa)
- `--freeze-layers N` — đóng băng embedding + N tầng dưới (chỉ train tầng trên +
  head). Backward rẻ hơn nhiều. Mặc định 8/12. Đặt 0 = fine-tune toàn bộ (chậm nhất).
- `--max-len 256` — abstract hiếm khi cần 320 token; ngắn hơn → rẻ hơn theo bình phương.
- `--batch-size` nhỏ + `--grad-accum` — batch hiệu dụng lớn mà RAM thấp.
- `--threads` — số nhân vật lý.
- `--bf16` — bật nếu CPU đời mới (AVX-512/AMX); tự bỏ qua nếu không hỗ trợ.
- `--model nlpie/distil-biobert` — model 6 tầng, nhanh gấp ~2–3 lần, độ chính xác giảm nhẹ.
- **Tăng tốc kỹ thuật sẵn có:** dynamic padding + length-grouped batching → giảm
  token thừa do padding (thường nhanh 2–4×).

> Mẹo: thử nhanh với `--limit 800` để kiểm tra pipeline chạy thông trước khi chạy full.

---

## 3. Sau khi chạy xong

Các PNG trong `experiments/results/` đã được cập nhật → mở report và thay hình,
chép số từ khối "COPY THESE INTO TABLE 5".

Nếu muốn so sánh nhiều baseline để dán bảng/biểu đồ:
```bat
python experiments\baselines\compare_models.py
python experiments\baselines\evaluate_heuristic.py
```

---

## Lưu ý quan trọng
- Số liệu phụ thuộc dữ liệu thật `data/processed/labeled_dataset.jsonl` (8.005 mẫu)
  và seed = 42. Mỗi máy/seed có thể lệch nhẹ ở chữ số thập phân cuối — đó là bình thường.
- Mình **không tạo số/chart giả**: phải chạy thật để có hình đúng cho khóa luận.
