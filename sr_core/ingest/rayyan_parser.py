import pandas as pd
import os

def parse_rayyan_csv(csv_path: str) -> dict:
    """
    Parses a Rayyan customizations_log.csv file.
    Rayyan stores actions as a timeline. This extracts the final label for each paper.
    Returns: dict mapping article_id (string) to their final state (int or str).
    """
    try:
        # Load the CSV, parsing dates
        df = pd.read_csv(csv_path, parse_dates=['created_at'])
        
        # Sort by created_at to ensure timeline is sequential
        df = df.sort_values(by='created_at')
        
        # Filter for 'included' keys.
        # Rayyan values are typically:
        # '1'   => Included
        # '-1'  => Excluded
        # '0'   => Uncertain/Maybe
        include_events = df[df['key'] == 'included'].copy()
        
        # Keep only the last action for each article_id
        final_labels_df = include_events.drop_duplicates(subset=['article_id'], keep='last')
        
        # Convert to dictionary { 'article_id': 'final_value'}
        final_labels = {}
        for _, row in final_labels_df.iterrows():
             final_labels[str(row['article_id'])] = str(row['value'])
             
        return final_labels
    except Exception as e:
        print(f"Error parsing Rayyan CSV {csv_path}: {e}")
        return {}

def merge_rayyan_export_folders(include_folder: str, exclude_folder: str):
    """
    Reads the data from both folders and combines them.
    Returns a unified mapping of article_id to their combined final label.
    """
    include_csv = os.path.join(include_folder, "customizations_log.csv")
    exclude_csv = os.path.join(exclude_folder, "customizations_log.csv")
    
    include_labels = parse_rayyan_csv(include_csv) if os.path.exists(include_csv) else {}
    exclude_labels = parse_rayyan_csv(exclude_csv) if os.path.exists(exclude_csv) else {}
    
    # Merge the dicts. If an ID exists in both (unlikely for final export, but possible),
    # we might need to compare timestamps. For now, since they are static exports,
    # we just merge them. 
    merged = {**exclude_labels, **include_labels}
    return merged
