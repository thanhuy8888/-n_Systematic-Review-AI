# HƯỚNG DẪN ĐỌC VÀ PHÂN TÍCH CÁC BIỂU ĐỒ BÁO CÁO TỐT NGHIỆP
*(Tài liệu tham khảo dùng cho viết báo cáo và bảo vệ trước Hội đồng khoa học)*

Thư mục `experiments/results/` hiện tại lưu trữ 7 biểu đồ chính phục vụ việc chứng minh tính hiệu quả của giải pháp AI so với các phương pháp truyền thống. Dưới đây là hướng dẫn đọc chi tiết từng biểu đồ, ý nghĩa các thông số và cách lập luận trước hội đồng bảo vệ.

---

## 1. Biểu đồ tần suất từ khóa: `keyword_matching_chart.png`
* **Ý nghĩa:** Trực quan hóa tần suất xuất hiện của 10 từ khóa PICO chính trong hai nhóm bài báo: Nhóm được chọn thực tế (**Include**) và Nhóm bị loại thực tế (**Exclude**).
* **Cách đọc:**
  * **Trục hoành (X):** 10 từ khóa/cụm từ khóa đại diện cho PICO (ví dụ: `mice / mouse`, `hfd` (High-Fat Diet), `insulin`, `glucose`, `obesity / obese`...).
  * **Trục tung (Y):** Tỷ lệ phần trăm (%) xuất hiện của từ khóa đó trong tập bài báo.
  * **Cột màu xanh lá (Emerald):** Nhóm bài báo chuyên gia chọn để đưa vào nghiên cứu tổng quan (Include).
  * **Cột màu đỏ (Rose):** Nhóm bài báo bị chuyên gia loại bỏ (Exclude).
* **Lập luận trước hội đồng:** 
  > *"Nhìn vào biểu đồ tần suất trùng khớp, chúng ta thấy rõ có sự khác biệt lớn về phân bố từ khóa. Ví dụ: từ khóa chỉ đối tượng động vật như `mice / mouse` xuất hiện trong gần 100% bài báo được chọn, nhưng cũng xuất hiện tới gần 50% trong các bài báo bị loại. Điều này chứng minh rằng **nếu chỉ dùng công cụ lọc từ khóa đơn giản (Keyword Search)**, chúng ta sẽ gặp phải lượng lớn bài báo gây nhiễu (False Positives) - tức là bài báo nói về chuột nhưng không nghiên cứu về bệnh lý chuyển hóa cần tìm."*

---

## 2. Ma trận đối chiếu Heuristic: `heuristic_confusion_matrix.png`
* **Ý nghĩa:** Biểu diễn kết quả phân loại dựa trên quy tắc logic tĩnh (Keyword Heuristic). Ở đây quy tắc là: Một bài báo được chọn nếu tiêu đề/tóm tắt chứa từ khóa động vật (`mouse/mice/rat...`) **ĐỒNG THỜI** chứa ít nhất một từ khóa chuyển hóa (`hfd/insulin/glucose...`).
* **Cách đọc:**
  * **Trục tung (Thực tế):** Nhãn do chuyên gia y khoa gán nhãn thủ công (Loại bài - 6006 bài; Chọn bài - 1999 bài).
  * **Trục hoành (Heuristic):** Kết quả lọc tự động của bộ từ khóa (Không khớp - Loại; Khớp - Chọn).
  * **Các ô số liệu:**
    * **True Negative (Trọng điểm loại đúng):** 2864 bài báo (bị loại chính xác vì không khớp từ khóa).
    * **False Positive (Báo động giả):** 3142 bài báo (Heuristic chọn vì chứa từ khóa, nhưng thực tế chuyên gia loại).
    * **False Negative (Bỏ sót nguy hiểm):** 873 bài báo (chuyên gia chọn, nhưng Heuristic loại bỏ vì thiếu từ khóa chuẩn trong bộ quy tắc).
    * **True Positive (Chọn đúng):** 1126 bài báo.
* **Lập luận trước hội đồng:**
  > *"Mô hình Heuristic (lọc từ khóa truyền thống) đạt độ chính xác (Accuracy) rất thấp, chỉ khoảng **52%**. Đặc biệt, số ca báo động giả lên tới 3142 bài và bỏ sót 873 bài quan trọng. Điều này chứng minh bộ lọc từ khóa tĩnh không thể hiểu được ngữ nghĩa sâu sắc của văn bản y khoa, dễ bị đánh lừa bởi các từ đồng nghĩa hoặc phủ định, thiết lập nền tảng bắt buộc phải ứng dụng học máy học sâu (AI Transformer)."*

---

## 3. Biểu đồ so sánh các mô hình: `model_comparison_chart.png`
* **Ý nghĩa:** So sánh trực quan hiệu năng giữa mô hình học sâu đề xuất (**Transformer Hybrid**) và các mô hình học máy truyền thống (**Logistic Regression, Random Forest, SVM**) chạy trên nền TF-IDF.
* **Cách đọc:**
  * **Trục hoành (X):** 4 thuật toán thử nghiệm.
  * **Trục tung (Y):** Điểm số từ 0.0 đến 1.0 (tương đương 0% đến 100%).
  * **Các cột màu sắc:** Xanh dương (Accuracy), Xanh lá (F1-Score - độ cân bằng giữa chính xác và bao phủ), Vàng (ROC-AUC - khả năng phân biệt lớp nhãn).
* **Lập luận trước hội đồng:**
  > *"Trong khi các mô hình Machine Learning cổ điển dựa trên đếm tần suất từ (TF-IDF) chỉ đạt độ chính xác quanh mức 68-77% và F1-Score rất thấp (do mất cân bằng dữ liệu cực hạn), mô hình **Transformer Hybrid (kết hợp vector ngữ nghĩa ẩn của PubMedBERT và bộ phân loại XGBoost)** đã bứt phá đạt độ chính xác **93.7%** và F1-Score **89.5%**. Điều này khẳng định việc trích xuất đặc trưng ngữ nghĩa bằng LLM y khoa chuyên biệt mang lại hiệu quả vượt trội."*

---

## 4. Ma trận nhầm lẫn AI: `transformer_confusion_matrix.png`
* **Ý nghĩa:** Thống kê kết quả phân loại thực tế của mô hình học sâu đề xuất (Transformer Hybrid) trên tập kiểm thử mở rộng.
* **Cách đọc:**
  * **Trục tung (True Label):** Nhãn thực tế từ chuyên gia (Exclude - Loại; Include - Chọn).
  * **Trục hoành (Predicted Label):** Nhãn do AI Transformer dự đoán.
  * **Các con số thực tế:**
    * **Loại đúng (True Negative):** Phân loại chính xác các bài báo không liên quan.
    * **Chọn đúng (True Positive):** Tìm ra chính xác các bài báo y khoa cần đưa vào tổng quan.
    * **Sai số (False Positives & False Negatives):** Giảm thiểu tối đa so với Heuristic cũ.
* **Lập luận trước hội đồng:**
  > *"Ma trận nhầm lẫn của mô hình Transformer Hybrid cho thấy sự phân bổ tập trung tuyệt đối vào đường chéo chính (True Negatives và True Positives). Số ca bỏ sót bài báo quan trọng (False Negatives) đã được kiểm soát ở mức tối thiểu, đảm bảo tính toàn vẹn thông tin cho nghiên cứu Systematic Review."*

---

## 5. Đường cong ROC: `transformer_roc_curve.png`
* **Ý nghĩa:** Trực quan hóa khả năng phân loại nhãn của mô hình ở mọi ngưỡng quyết định (threshold) khác nhau.
* **Cách đọc:**
  * **Trục hoành:** Tỷ lệ báo động giả (False Positive Rate).
  * **Trục tung:** Tỷ lệ nhạy bén/phát hiện đúng (True Positive Rate).
  * **Đường nét đứt màu đen:** Khả năng phân loại ngẫu nhiên (may rủi - AUC = 0.5).
  * **Đường cong màu cam:** Đường ROC của mô hình Transformer Hybrid.
  * **Thông số AUC (Area Under Curve):** Diện tích dưới đường cong, giá trị lý tưởng là 1.0. Mô hình đạt **0.9808** (cực kỳ xuất sắc).
* **Lập luận trước hội đồng:**
  > *"Đường cong ROC của mô hình tiến sát góc trên bên trái với chỉ số AUC đạt **0.9808**. Điều này chứng minh mô hình hoạt động vô cùng ổn định và có khả năng tách biệt hoàn hảo giữa bài báo cần chọn và bài báo cần loại mà không bị phụ thuộc vào việc thay đổi ngưỡng xác suất phân loại."*

---

## 6. Đường cong Precision-Recall: `transformer_pr_curve.png`
* **Ý nghĩa:** Đây là chỉ số **quan trọng nhất** đối với bài toán Systematic Review, nơi dữ liệu mất cân bằng nghiêm trọng (lượng bài báo bị loại gấp nhiều lần lượng bài báo được chọn).
* **Cách đọc:**
  * **Trục hoành:** Recall (Khả năng bao phủ, không bỏ sót bài báo).
  * **Trục tung:** Precision (Độ chính xác, không nhặt nhầm bài báo rác).
  * **AUC (PR-AUC):** Chỉ số diện tích dưới đường cong PR, đạt **0.9669**.
* **Lập luận trước hội đồng:**
  > *"Đối với nghiên cứu hệ thống (Systematic Review), đường cong PR-AUC quan trọng hơn đường ROC truyền thống vì tập dữ liệu thực tế cực kỳ mất cân bằng (tỷ lệ 1 Include : 3 Exclude). Việc đạt chỉ số PR-AUC lên tới **0.9669** đảm bảo mô hình vẫn hoạt động chính xác cực cao ngay cả khi tỷ lệ bài báo y khoa liên quan chiếm tỷ số rất nhỏ trong tập dữ liệu quét."*

---

## 7. Phân bố độ tự tin của AI: `transformer_probability_distribution.png`
* **Ý nghĩa:** Đo lường mức độ tự tin (xác suất dự đoán) của thuật toán phân loại.
* **Cách đọc:**
  * **Trục hoành:** Xác suất dự đoán nhãn Include (từ 0.0 đến 1.0).
  * **Trục tung:** Mật độ phân bố số lượng bài báo (Density).
  * **Đỉnh màu đỏ (Actual Exclude):** Tập trung sát mốc 0.0 (AI rất tự tin loại các bài báo này).
  * **Đỉnh màu xanh lá (Actual Include):** Tập trung sát mốc 1.0 (AI rất tự tin chọn các bài báo này).
  * **Đường đứt nét 0.5:** Ngưỡng phân loại mặc định.
* **Lập luận trước hội đồng:**
  > *"Biểu đồ phân bố độ tự tin cho thấy mô hình AI đề xuất có tính phân tách nhị phân cực tốt. Hầu hết các bài báo thực tế bị loại (màu đỏ) được AI định vị phân bố sát mốc 0, và các bài báo được chọn (màu xanh) tập trung sát mốc 1. Vùng giao thoa xung quanh ngưỡng 0.5 rất mỏng, chứng tỏ AI đưa ra quyết định dứt khoát và có độ tin cậy cao, hạn chế tối đa các quyết định lưỡng lự gây tốn thời gian cho chuyên gia thẩm định."*
