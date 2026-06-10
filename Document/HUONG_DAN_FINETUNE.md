# Hướng dẫn fine-tune PubMedBERT để cải thiện độ chính xác screening

Tài liệu này mô tả hướng "cải thiện mạnh" cho mô-đun screening: thay vì *đóng băng*
PubMedBERT rồi xếp XGBoost lên trên các embedding tĩnh (cấu hình cũ trong
`evaluate_screening_clean.py`), ta **fine-tune toàn bộ PubMedBERT end-to-end** như
một bộ phân loại nhị phân. Cách này cho phép các tầng ngữ cảnh thích nghi với ranh
giới include/exclude và thường nâng ROC-AUC lên rõ rệt so với baseline embedding tĩnh.

## 1. Vì sao cách này tốt hơn

- **Fine-tune end-to-end:** cập nhật trọng số của chính PubMedBERT theo nhãn của bài
  toán, thay vì chỉ học một bộ phân loại nông trên đặc trưng cố định.
- **Bỏ SMOTE, dùng class-weight:** SMOTE nội suy điểm giả trong không gian 768 chiều
  → không đáng tin. Thay bằng trọng số lớp trong hàm mất mát (CrossEntropyLoss) là
  cách chuẩn và trung thực để xử lý mất cân bằng 1:3.
- **Giữ quy trình leakage-free:** chia 70/15/15 (train/val/test) **một lần**, tập test
  được giữ riêng, không dùng để tune hay chọn ngưỡng.
- **Chọn ngưỡng đúng bài toán SR:** dò ngưỡng trên tập **validation** cho 2 điểm vận
  hành: (a) F1 tối ưu, (b) recall ≥ 95% → báo cáo **WSS@95** (Work-Saved-over-Sampling),
  rồi áp nguyên ngưỡng đó lên tập test.

## 2. Yêu cầu

- **GPU NVIDIA (CUDA)** — rất khuyến nghị. Trên CPU sẽ chạy hàng giờ.
- Không cần cài thêm thư viện: `torch`, `transformers`, `scikit-learn`, `matplotlib`,
  `seaborn` đã có trong `requirements.txt`.
- Dữ liệu: `data/processed/labeled_dataset.jsonl` (đã có sẵn, 8005 bài).

## 3. Cách chạy

Từ thư mục gốc dự án, đã kích hoạt venv:

```bash
python experiments/baselines/finetune_pubmedbert.py
```

Tùy chọn:

```bash
python experiments/baselines/finetune_pubmedbert.py \
    --model microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext \
    --epochs 4 --batch-size 16 --lr 2e-5 --max-len 320 --target-recall 0.95
```

Mô hình thay thế đáng thử: `--model michiyasunaga/BioLinkBERT-base`.

Lần chạy đầu sẽ tải checkpoint từ HuggingFace (~440MB), cần mạng.

## 4. Script sẽ tạo ra gì

- `sr_core/screening_model/finetuned_pubmedbert/` — model + tokenizer đã fine-tune.
- `sr_core/screening_model/finetuned_meta.json` — ngưỡng + toàn bộ metric trên tập test.
- Ghi đè các hình **thật** (leakage-free) vào `experiments/results/`:
  `transformer_roc_curve.png`, `transformer_pr_curve.png`,
  `transformer_confusion_matrix.png`, `transformer_probability_distribution.png`.
- In ra khối **"COPY THESE INTO TABLE 5"** với Precision / Recall / F1 / ROC-AUC /
  PR-AUC / WSS@95 mới.

## 5. Hệ thống tự dùng model mới (không cần sửa code khác)

`sr_core/screening_model/transformer_screen.py` đã được cập nhật với thứ tự ưu tiên:

```
finetuned  ->  hybrid (XGBoost)  ->  embedding-only  ->  keyword heuristic
```

Khi thư mục `finetuned_pubmedbert/` tồn tại, API sẽ tự động dùng nó (mode
`finetuned`) và áp ngưỡng đã tune trong `finetuned_meta.json`. Nếu xóa thư mục đó,
hệ thống quay lại đúng hành vi cũ — **tương thích ngược hoàn toàn**.

## 6. Sau khi chạy: cập nhật khóa luận

1. Lấy các số trong khối "COPY THESE INTO TABLE 5" → cập nhật **Bảng 5** và đoạn văn
   mô tả screening (đoạn 178 trong báo cáo).
2. Bổ sung cột/điểm **WSS@95** — đây là metric chuẩn của ngành SR, làm nổi bật đóng góp.
3. Các hình ROC/PR/Confusion/Probability trong `experiments/results/` đã được vẽ lại
   bằng số thật → thay vào báo cáo (đảm bảo chart khớp Bảng 5, tránh "tham chart sai
   bản chất").
4. Ghi rõ trong phần phương pháp: *fine-tune end-to-end, class-weight thay SMOTE,
   split 70/15/15 leakage-free, ngưỡng tune trên validation*.

## 7. Lưu ý quan trọng

- Model fine-tune học **đúng tiêu chí review của bộ dữ liệu này** (rối loạn chuyển hóa
  ở chuột). Đây là cách tiếp cận hợp lý cho case study của khóa luận, nhưng cần nói rõ
  rằng để áp cho một review khác thì phải fine-tune lại trên nhãn của review đó (hoặc
  dùng hướng active learning). Đây cũng là một điểm bàn luận tốt cho phần "future work".
- Đặt `--seed` cố định (mặc định 42) để kết quả tái lập được khi bảo vệ.
