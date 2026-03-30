import os
import json
import logging
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sr_core.ingest.parser import parse_ris
from sr_core.ingest.rayyan_parser import parse_rayyan_csv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_label_from_value(val: str) -> str:
    """Map Rayyan values back to our standard string labels"""
    # 1 -> include, -1 -> exclude, 0 -> uncertain
    if val == '1' or val == 1:
        return 'include'
    elif val == '-1' or val == -1:
        return 'exclude'
    else:
        return 'uncertain'

def process_exported_rayyan_dataset(raw_data_path: str, output_path: str):
    """
    Parses Rayyan raw exports from both Include and Exclude directories.
    Combines the .ris datasets and joins the final human labeling decision.
    """
    include_dir = os.path.join(raw_data_path, "Include")
    exclude_dir = os.path.join(raw_data_path, "Exclude")
    
    all_papers = []
    
    for tag_dir in [include_dir, exclude_dir]:
        if not os.path.exists(tag_dir):
            logger.warning(f"Directory {tag_dir} not found. Skipping.")
            continue
            
        ris_path = os.path.join(tag_dir, "articles.ris")
        csv_path = os.path.join(tag_dir, "customizations_log.csv")
        
        # 1. Parse texts
        if os.path.exists(ris_path):
            logger.info(f"Parsing RIS file: {ris_path}")
            papers = parse_ris(ris_path)
        else:
            logger.warning(f"articles.ris not found in {tag_dir}")
            papers = []
            
        # 2. Parse labels
        if os.path.exists(csv_path):
            logger.info(f"Parsing CSV Labels: {csv_path}")
            labels_map = parse_rayyan_csv(csv_path)
        else:
            logger.warning(f"customizations_log.csv not found in {tag_dir}")
            labels_map = {}
            
        # 3. Merge
        for p in papers:
            # Note: The RIS parse logic created a `paper_id` based on Hash.
            # Rayyan uses its internal `article_id` in the CSV, but unfortunately
            # the RIS file from Rayyan export does NOT guarantee the `ID` field exists or matches.
            # *Assumption*: Since the folders are ALREADY split perfectly into "Include" and "Exclude",
            # we can use the folder name as the fallback ground truth label!
            folder_name = os.path.basename(tag_dir)
            p['human_label'] = 'include' if folder_name.lower() == 'include' else 'exclude'
            
            all_papers.append(p)
            
    # Write to JSONL
    logger.info(f"Writing {len(all_papers)} papers to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for p in all_papers:
            f.write(json.dumps(p) + "\n")
            
    logger.info("Dataset ingestion complete.")
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest Rayyan Data")
    parser.add_argument("--raw", type=str, required=True, help="Path to raw data containing Include/Exclude folders")
    parser.add_argument("--out", type=str, required=True, help="Path to output labeled dataset (JSONL)")
    args = parser.parse_args()
    process_exported_rayyan_dataset(args.raw, args.out)
