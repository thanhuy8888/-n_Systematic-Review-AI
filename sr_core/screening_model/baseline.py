import os
import json
import numpy as np
import re
import nltk
from nltk.stem import WordNetLemmatizer
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        tokens = re.findall(r'\b\w\w+\b', doc)
        return [self.wnl.lemmatize(t) for t in tokens]

class BaselineModel:
    def __init__(self):
        # Update Vectorizer: Tăng số lượng từ, xét cả cụm 3 từ (trigram)
        self.vectorizer = TfidfVectorizer(
            max_features=10000, 
            stop_words='english', 
            ngram_range=(1, 3),
            tokenizer=LemmaTokenizer(),
            token_pattern=None
        )
        
        # Thiết lập Khối thuật toán:
        clf1 = LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42)
        clf2 = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
        clf3 = SVC(kernel='linear', class_weight='balanced', probability=True, random_state=42)
        
        # Kết hợp cả 3 thuật toán bằng Voting Soft:
        self.clf = VotingClassifier(
            estimators=[('lr', clf1), ('rf', clf2), ('svc', clf3)],
            voting='soft'
        )
    
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
        
        print("Applying SMOTE to balance the dataset...")
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        print(f"After SMOTE: Includes: {sum(y_train_res == 1)}, Excludes: {sum(y_train_res == 0)}")
        
        print("Training Ensemble Classifier...")
        self.clf.fit(X_train_res, y_train_res)
        
        # Evaluation
        print("\n--- Evaluation on Test Set ---")
        y_pred = self.clf.predict(X_test)
        y_prob = self.clf.predict_proba(X_test)[:, 1]
        
        print(classification_report(y_test, y_pred, target_names=["exclude", "include"]))
        
        auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        
        print(f"ROC-AUC = {auc:.4f}")
        print(f"PR-AUC  = {pr_auc:.4f} (Crucial for Systematic Reviews)")
        
        try:
            self.plot_results(y_test, y_pred, y_prob)
        except Exception as e:
            print(f"Warning: Failed to generate plots: {e}")
        
        return {
            "auc": auc,
            "pr_auc": pr_auc
        }
        
    def plot_results(self, y_test, y_pred, y_prob):
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
        
        output_dir = os.path.join("experiments", "results")
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Confusion Matrix
        plt.figure(figsize=(6, 5))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Exclude', 'Include'], yticklabels=['Exclude', 'Include'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
        plt.close()
        
        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_score(y_test, y_prob):.3f})')
        plt.plot([0.0, 1.0], [0.0, 1.0], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "roc_curve.png"))
        plt.close()
        
        # 3. Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        plt.figure(figsize=(6, 5))
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {average_precision_score(y_test, y_prob):.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "pr_curve.png"))
        plt.close()
        
        print(f"Plots have been successfully saved to '{output_dir}'")

    def predict(self, text: str):
        if not hasattr(self.vectorizer, 'vocabulary_'):
             raise ValueError("Model is not trained yet.")
        
        X = self.vectorizer.transform([text])
        prob = self.clf.predict_proba(X)[0, 1]
        return prob
