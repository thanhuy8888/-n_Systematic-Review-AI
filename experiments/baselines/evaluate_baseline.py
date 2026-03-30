import os
import sys
import json

# Add root project path explicitly
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sr_core.screening_model.baseline import BaselineModel

def load_data(jsonl_path: str) -> list[dict]:
    papers = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            papers.append(json.loads(line))
    return papers

if __name__ == "__main__":
    dataset_path = os.path.join("data", "processed", "labeled_dataset.jsonl")
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}.")
        print("Please run scripts/ingest_rayyan_data.py first.")
        sys.exit(1)
        
    papers = load_data(dataset_path)
    
    model = BaselineModel()
    results = model.train_and_eval(papers)
    
    print("\n[Done] Baseline evaluation complete.")
