import os
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

class BaselineModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
        # Handle class imbalance using class_weight='balanced'
        self.clf = LogisticRegression(class_weight='balanced', max_iter=1000)
    
    def prepare_data(self, papers: list[dict]):
        """Extract texts and labels."""
        X_raw = []
        y = []
        
        for p in papers:
            text = f"{p.get('title', '')} {p.get('abstract', '')}"
            label = 1 if p.get('human_label') == 'include' else 0
            
            X_raw.append(text)
            y.append(label)
        
        return X_raw, np.array(y)

    def train_and_eval(self, papers: list[dict], test_size=0.2, random_state=42):
        """Train the TF-IDF+Logistic baseline and print benchmark metrics."""
        
        X_raw, y = self.prepare_data(papers)
        
        if len(set(y)) < 2:
            print("Error: Dataset must contain both 'include' and 'exclude' labels.")
            return
        
        print(f"Total papers: {len(X_raw)}")
        print(f"Includes: {sum(y == 1)}, Excludes: {sum(y == 0)}")
        
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_raw, y, test_size=test_size, stratify=y, random_state=random_state
        )
        
        print("Vectorizing text using TF-IDF...")
        X_train = self.vectorizer.fit_transform(X_train_raw)
        X_test = self.vectorizer.transform(X_test_raw)
        
        print("Training Logistic Regression...")
        self.clf.fit(X_train, y_train)
        
        # Evaluation
        print("\n--- Evaluation on Test Set ---")
        y_pred = self.clf.predict(X_test)
        y_prob = self.clf.predict_proba(X_test)[:, 1]
        
        print(classification_report(y_test, y_pred, target_names=["exclude", "include"]))
        
        auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        
        print(f"ROC-AUC = {auc:.4f}")
        print(f"PR-AUC  = {pr_auc:.4f} (Crucial for Systematic Reviews)")
        
        return {
            "auc": auc,
            "pr_auc": pr_auc
        }
        
    def predict(self, text: str):
        if not hasattr(self.vectorizer, 'vocabulary_'):
             raise ValueError("Model is not trained yet.")
        
        X = self.vectorizer.transform([text])
        prob = self.clf.predict_proba(X)[0, 1]
        return prob
