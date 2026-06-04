import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Set encoding to utf-8
sys.stdout.reconfigure(encoding='utf-8')

def load_data(jsonl_path):
    papers = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            papers.append(json.loads(line))
    return papers

def main():
    dataset_path = os.path.join("data", "processed", "labeled_dataset.jsonl")
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        sys.exit(1)
        
    print("Đang tải dữ liệu gán nhãn thực tế...")
    papers = load_data(dataset_path)
    total_papers = len(papers)
    print(f"Tổng số bài báo loaded: {total_papers}")
    
    y_true = []
    y_pred = []
    
    # Danh sách từ khóa PICO để tính toán biểu đồ tần suất trùng khớp
    keywords_to_track = [
        "mice / mouse", "hfd", "high fat", "insulin", "glucose", 
        "obesity / obese", "cholesterol", "triglyceride", "steatosis", "homa-ir"
    ]
    
    # Map để đếm số lần khớp trong nhóm Include vs Exclude
    kw_counts_include = {kw: 0 for kw in keywords_to_track}
    kw_counts_exclude = {kw: 0 for kw in keywords_to_track}
    
    n_includes = 0
    n_excludes = 0
    
    # Thiết lập hạt giống ngẫu nhiên để thống nhất kết quả chạy đẹp mắt cho đồ án
    np.random.seed(42)
    
    print("Đang chạy đối chiếu mô hình lọc Heuristic ngữ nghĩa PICO trên toàn tập dữ liệu...")
    for p in papers:
        title = p.get("title", "")
        abstract = p.get("abstract", "")
        text = f"{title}. {abstract}".lower()
        
        # Nhãn gốc từ chuyên gia
        true_label = 1 if p.get("human_label") == "include" else 0
        y_true.append(true_label)
        
        if true_label == 1:
            n_includes += 1
        else:
            n_excludes += 1
            
        # Thống kê từ khóa xuất hiện thực tế (100% Real data)
        for kw in keywords_to_track:
            parts = [part.strip() for part in kw.split("/")]
            matched = any(part in text for part in parts)
            if matched:
                if true_label == 1:
                    kw_counts_include[kw] += 1
                else:
                    kw_counts_exclude[kw] += 1
                    
        # Phân loại thực tế bằng luật Heuristic PICO Matching
        # Quy tắc: Văn bản chứa từ khóa động vật (mouse/mice) VÀ ít nhất một từ khóa chỉ chế độ ăn/chuyển hóa khác
        has_animal = any(w in text for w in ["mice", "mouse", "murine", "rat", "animal"])
        has_other_kw = False
        for kw in keywords_to_track[1:]:
            parts = [part.strip() for part in kw.split("/")]
            if any(part in text for part in parts):
                has_other_kw = True
                break
        
        pred_label = 1 if (has_animal and has_other_kw) else 0
        y_pred.append(pred_label)
        
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Tính toán Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    print("\n" + "="*50)
    print("📌 KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH HÌNH THỨC PICO (HEURISTIC BASELINE)")
    print("="*50)
    print(classification_report(y_true, y_pred, target_names=["Exclude", "Include"]))
    
    # 3. Vẽ biểu đồ Confusion Matrix thực tế
    output_dir = os.path.join("experiments", "results")
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(6, 5))
    sns.set_theme(style="white")
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Loại (Không khớp)', 'Chọn (Khớp)'], 
                yticklabels=['Loại (Thực tế)', 'Chọn (Thực tế)'],
                annot_kws={"size": 14, "weight": "bold"})
    plt.title('Ma trận đối chiếu thực tế (Heuristic PICO Matching)', fontsize=13, pad=15, weight='bold')
    plt.ylabel('Nhãn Thực Tế (Chuyên gia)', fontsize=12, labelpad=10)
    plt.xlabel('Nhãn lọc từ khóa (Heuristic)', fontsize=12, labelpad=10)
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "heuristic_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[Đã xuất biểu đồ] Confusion Matrix thực tế: {cm_path}")
    
    # 4. Vẽ biểu đồ Tần suất Từ khóa trùng khớp (Keyword Matching Frequency Graph)
    p_includes = [kw_counts_include[kw] / max(1, n_includes) * 100 for kw in keywords_to_track]
    p_excludes = [kw_counts_exclude[kw] / max(1, n_excludes) * 100 for kw in keywords_to_track]
    
    x = np.arange(len(keywords_to_track))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    rects1 = ax.bar(x - width/2, p_includes, width, label='Included Papers (Chọn)', color='#10b981') # Emerald
    rects2 = ax.bar(x + width/2, p_excludes, width, label='Excluded Papers (Loại)', color='#f43f5e') # Rose
    
    ax.set_ylabel('Tỷ lệ xuất hiện trong nhóm (%)', fontsize=12, labelpad=10, weight='bold')
    ax.set_title('Biểu đồ Tần suất Trùng khớp Từ khóa PICO (Keyword Matching Frequency)', fontsize=14, pad=20, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(keywords_to_track, rotation=25, ha="right", fontsize=10, weight='bold')
    ax.legend(fontsize=11)
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, weight='bold')
            
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    kw_path = os.path.join(output_dir, "keyword_matching_chart.png")
    plt.savefig(kw_path, dpi=300)
    plt.close()
    print(f"[Đã xuất biểu đồ] Tần suất Từ khóa PICO: {kw_path}")
    
    print("\n[Hoàn tất] Sinh biểu đồ thực tế và trực quan từ khóa thành công!")

if __name__ == "__main__":
    main()
