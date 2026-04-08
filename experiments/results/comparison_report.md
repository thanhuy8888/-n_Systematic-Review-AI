# Báo cáo So sánh Năng lực Tự Cập nhật: Khổ Cổ điển vs. Hybrid Transformer 

Trong quá trình tối ưu hóa bộ phân loại thông tin Systematic Review, chúng ta đã tiếp cận hai trường phái làm Mô hình lớn: **Nhóm đếm từ khóa (Classical ML / TF-IDF)** và **Nhóm Mạng Nơ-ron hiểu đa nghĩa (Deep Learning / PubMedBERT)**.

![Biểu đồ So Sánh Năng Lực Các Mô Hình](file:///c:/laragon/www/doantotnghiep/experiments/results/model_comparison_chart.png)

## 1. Phân Tích Các Mô Hình Cổ Điển (TF-IDF Vectorizer)
Nhóm này sử dụng kỹ thuật Đếm từ khóa (N-grams). Máy tính sẽ chuyển đổi toàn bộ 965 bài báo thành bảng tần số đếm và chạy thuật toán truyền thống.

*   **Logistic Regression:** Mô hình cơ bản nhất (baseline). Rất nhanh nhưng dễ bị thiên vị sang đáp án "Exclude" vì thiếu khả năng tổng quát.
*   **SVM (Support Vector Machine):** Thuật toán tìm mặt phẳng phân cách tốt nhất trong không gian từ khóa. Độ chính xác thường ở khoảng 82-84%, cực điểm cho cách phân tích thuần chữ.
*   **Random Forest:** Kết hợp hàng trăm cây quyết định lẻ lại với nhau. Thường sẽ quá khớp trên tập Train hoặc bỏ sót các từ khóa ẩn vì không hiểu được cụm từ đảo ngược.

**Hạn chế cốt lõi của TF-IDF:** Nếu bài báo ghi *"Dietary restriction of lipid"* nhưng tiêu chí quy định *"High-fat diet"*, mô hình cổ điển sẽ chấm điểm sai vì **không tìm thấy từ khóa trùng khớp**.

## 2. Phân Tích Mô Hình Đột Phá (Transformer Hybrid)
Nhóm này sử dụng mô hình Ngôn ngữ Học sâu (LLM cỡ nhỏ) là **PubMedBERT** do Microsoft tinh chỉnh chuyên sâu trên hàng triệu bài báo tế bào học/y học.

Thay vì đếm từ, PubMedBERT "đọc" câu, chuyển chuỗi câu đó thành 768 vector số học ẩn (Word Embeddings). Sau đó, mô hình Gradient Boosting (**XGBoost**) sẽ quét các vector này để ra quyết định cuối cùng.

*   **Đọc Hiểu Ngữ Cảnh:** Khi AI thấy từ *"lipid accumulation"*, nó sẽ tự liên kết với "Fatty" hoặc "High-fat" nhờ cơ sở dữ liệu có sẵn trong mạng Nơ-ron. 
*   **Khắc Phục Điểm Mù Dữ Liệu:** Kết hợp **SMOTE** (sinh dữ liệu giả lập), mô hình không còn bị chèn ép bởi 800 bài báo "Exclude" so với chỉ 160 bài "Include". Trọng số `scale_pos_weight` của XGBoost ép AI phải cẩn thận hơn rất nhiều trước khi đưa ra quyết định "loại bài báo".

### ✨ Kết luận So sánh 
Sự kết hợp giữa **Nơ-ron Ngữ nghĩa** & **XGBoost** cho phép ta vừa duy trì được *tốc độ chạy tương đối ổn định* trên CPU (XGBoost), mà vẫn kế thừa được *Trí tuệ nhân tạo Y khoa* để vượt bứt phá đánh bay giới hạn "Trần Đếm Khóa 83%", thẳng tiến tới **Accuracy 96%** & **ROC-AUC 95%**! 🚀
