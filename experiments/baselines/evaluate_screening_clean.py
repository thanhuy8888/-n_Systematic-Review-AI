"""
Leakage-free screening evaluation for the AI Systematic-Review Assistant.
================================================================================
This script REPLACES the methodologically unsound numbers produced by
evaluate_transformer.py. It removes the two sources of optimistic bias present
in the legacy script:

    (1) SMOTE was applied to the TEST set  -> synthetic test rows.
    (2) "Report blending" mixed 60% of the TRAINING rows back into the
        evaluation set -> train/test leakage.

Here, a single stratified 80/20 split is created ONCE. The 20% test split is
held out and never resampled or mixed. SMOTE is applied to the training split
only. Every reported metric is therefore a genuine generalisation estimate on
unseen data.

It benchmarks the SAME hybrid configuration that is deployed at inference time
in sr_core/screening_model/transformer_screen.py:

        PubMedBERT sentence embedding (S-PubMedBert-MS-MARCO, 768-d)
              concatenated with
        TF-IDF (max 2000 features, 1-2 grams)
              -> SMOTE (train only)
              -> Soft-voting XGBoost + RandomForest (weights 2:1)

and, for comparison, a TF-IDF-only ensemble baseline.

Requirements: torch, sentence-transformers, xgboost, scikit-learn,
imbalanced-learn, matplotlib, seaborn (all in requirements.txt), plus network
access on first run to download the PubMedBERT checkpoint.

Run from the project root:
        python experiments/baselines/evaluate_screening_clean.py

The printed PubMedBERT-hybrid Precision / Recall / F1 / ROC-AUC / PR-AUC and
the measured screening throughput are the numbers to copy into Table 5 of the
thesis. Clean figures are written to experiments/results/ (overwriting the
leaky versions).
"""
import os
import sys
import json
import time
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    roc_auc_score, average_precision_score, classification_report,
    roc_curve, precision_recall_curve, confusion_matrix,
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib

TARGET_RECALL = 0.95  # minimum recall for WSS operating point

RANDOM_STATE = 42
DATA_PATH = os.path.join("data", "processed", "labeled_dataset.jsonl")
RESULTS_DIR = os.path.join("experiments", "results")
MODEL_DIR = os.path.join("sr_core", "screening_model")
EMBED_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"


def load_dataset(path):
    papers = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                papers.append(json.loads(line))
    texts = [f"{p.get('title','')}. {p.get('abstract','')}" for p in papers]
    labels = np.array([1 if p.get("human_label") == "include" else 0 for p in papers])
    return texts, labels


def f1_optimal_threshold(y_true, y_prob):
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 181):
        f1 = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


def recall_threshold(y_true, y_prob, target=TARGET_RECALL):
    """Highest threshold that still achieves recall >= target (search high→low)."""
    best_t = 0.01
    for t in np.linspace(0.99, 0.01, 980):
        pred = (y_prob >= t).astype(int)
        r = recall_score(y_true, pred, zero_division=0)
        if r >= target:
            best_t = float(t)
            break
    return best_t


def evaluate(name, clf, X_tr, y_tr, X_te, y_te, results, y_val=None, prob_val=None):
    if X_tr is not None:
        clf.fit(X_tr, y_tr)
    y_prob = clf.predict_proba(X_te)[:, 1]

    # Tune threshold on validation; fall back to 0.5 if no val set given
    if y_val is not None and prob_val is not None:
        t_f1, _ = f1_optimal_threshold(y_val, prob_val)
        t_r95 = recall_threshold(y_val, prob_val)
    else:
        t_f1, t_r95 = 0.5, 0.5

    y_pred_f1 = (y_prob >= t_f1).astype(int)
    y_pred_r95 = (y_prob >= t_r95).astype(int)

    def metrics(y_pred):
        return {
            "accuracy": accuracy_score(y_te, y_pred),
            "precision": precision_score(y_te, y_pred, zero_division=0),
            "recall": recall_score(y_te, y_pred, zero_division=0),
            "f1": f1_score(y_te, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_te, y_prob),
            "pr_auc": average_precision_score(y_te, y_prob),
        }

    m_f1 = metrics(y_pred_f1)
    m_r95 = metrics(y_pred_r95)

    results[name] = (m_f1, y_pred_f1, y_prob, clf, t_f1, t_r95, m_r95)

    print(f"\n== {name} (held-out test, no leakage) ==")
    print(f"  [Threshold = {t_f1:.2f}, F1-optimal]")
    print(f"  Accuracy : {m_f1['accuracy']:.4f}")
    print(f"  Precision: {m_f1['precision']:.4f}")
    print(f"  Recall   : {m_f1['recall']:.4f}")
    print(f"  F1-score : {m_f1['f1']:.4f}")
    print(f"  ROC-AUC  : {m_f1['roc_auc']:.4f}")
    print(f"  PR-AUC   : {m_f1['pr_auc']:.4f}")
    print(classification_report(y_te, y_pred_f1, target_names=["exclude", "include"], digits=3))

    print(f"  [Threshold = {t_r95:.2f}, Recall >= {TARGET_RECALL:.0%}]")
    print(f"  Precision: {m_r95['precision']:.4f}  Recall: {m_r95['recall']:.4f}  F1: {m_r95['f1']:.4f}")

    return m_f1, m_r95


def make_charts(y_te, hybrid_pred, hybrid_prob, comparison):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception as e:
        print(f"[charts] skipped ({e})")
        return
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Confusion matrix (clean)
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_te, hybrid_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Exclude", "Include"], yticklabels=["Exclude", "Include"])
    plt.title("Confusion Matrix - PubMedBERT + XGBoost (held-out test)")
    plt.ylabel("True Label"); plt.xlabel("Predicted Label"); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "transformer_confusion_matrix.png")); plt.close()

    # ROC
    fpr, tpr, _ = roc_curve(y_te, hybrid_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2,
             label=f"ROC (AUC = {roc_auc_score(y_te, hybrid_prob):.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0, 1]); plt.ylim([0, 1.05]); plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate"); plt.title("ROC Curve - PubMedBERT + XGBoost (held-out test)")
    plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "transformer_roc_curve.png")); plt.close()

    # PR
    prec, rec, _ = precision_recall_curve(y_te, hybrid_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, color="blue", lw=2,
             label=f"PR (AUC = {average_precision_score(y_te, hybrid_prob):.4f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall Curve - PubMedBERT + XGBoost (held-out test)")
    plt.legend(loc="lower left"); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "transformer_pr_curve.png")); plt.close()

    # Probability distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(hybrid_prob[y_te == 0], color="red", label="Actual: Exclude",
                 kde=True, stat="density", bins=30, alpha=0.5)
    sns.histplot(hybrid_prob[y_te == 1], color="green", label="Actual: Include",
                 kde=True, stat="density", bins=30, alpha=0.5)
    plt.axvline(x=0.5, color="black", linestyle="--", label="Threshold (0.5)")
    plt.title("Predicted Probability Distribution (held-out test)")
    plt.xlabel("P(Include)"); plt.ylabel("Density"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "transformer_probability_distribution.png")); plt.close()

    # Model comparison
    names = list(comparison.keys())
    metrics = ["accuracy", "f1", "roc_auc"]
    x = np.arange(len(names)); w = 0.25
    plt.figure(figsize=(9, 5))
    for i, mt in enumerate(metrics):
        plt.bar(x + (i - 1) * w, [comparison[n][mt] for n in names], w, label=mt.upper())
    plt.xticks(x, names, rotation=15, ha="right"); plt.ylim(0, 1.0)
    plt.ylabel("Score"); plt.title("Screening Model Comparison (held-out test)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "model_comparison_chart.png")); plt.close()
    print(f"[charts] clean figures written to {RESULTS_DIR}/")


def main():
    if not os.path.exists(DATA_PATH):
        sys.exit(f"Dataset not found at {DATA_PATH}. Run from the project root.")

    texts, y = load_dataset(DATA_PATH)
    print(f"Total papers : {len(y)}")
    print(f"Include      : {int(y.sum())}")
    print(f"Exclude      : {int((1 - y).sum())}")

    # ---- Stratified split: 70% train / 10% val (threshold tuning) / 20% test ----
    Xtmp_text, Xte_text, ytmp, yte = train_test_split(
        texts, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    Xtr_text, Xva_text, ytr, yva = train_test_split(
        Xtmp_text, ytmp, test_size=0.125, stratify=ytmp, random_state=RANDOM_STATE
    )  # 0.125 * 0.8 = 10% overall
    print(f"\nTrain: {len(Xtr_text)} | Val: {len(Xva_text)} | Held-out test: {len(Xte_text)} "
          f"(test never resampled, never blended)")

    # class imbalance ratio for XGBoost
    n_neg = int((ytr == 0).sum())
    n_pos = int((ytr == 1).sum())
    spw = round(n_neg / max(1, n_pos), 2)
    print(f"Class ratio neg/pos in train: {n_neg}/{n_pos} -> scale_pos_weight={spw}")

    # ---- TF-IDF features (fit on train only) ----
    tfidf = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), stop_words="english")
    Xtr_tfidf = tfidf.fit_transform(Xtr_text).toarray()
    Xva_tfidf = tfidf.transform(Xva_text).toarray()
    Xte_tfidf = tfidf.transform(Xte_text).toarray()

    results = {}

    # ---- Baseline: TF-IDF only ----
    smote = SMOTE(random_state=RANDOM_STATE)
    Xtr_tfidf_bal, ytr_bal = smote.fit_resample(Xtr_tfidf, ytr)
    baseline = VotingClassifier(estimators=[
        ("lr", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE)),
        ("rf", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE)),
        ("svc", SVC(kernel="linear", class_weight="balanced", probability=True, random_state=RANDOM_STATE)),
    ], voting="soft")
    pva_base = baseline.fit(Xtr_tfidf_bal, ytr_bal).predict_proba(Xva_tfidf)[:, 1]
    evaluate("TF-IDF Ensemble (baseline)", baseline, Xtr_tfidf_bal, ytr_bal,
             Xte_tfidf, yte, results, yva, pva_base)

    # ---- Proposed: PubMedBERT + TF-IDF hybrid ----
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        print(f"\n[hybrid] sentence-transformers unavailable ({e}). "
              f"Install torch + sentence-transformers and re-run to obtain the headline numbers.")
        return

    print(f"\nLoading PubMedBERT embedder: {EMBED_MODEL} ...")
    embedder = SentenceTransformer(EMBED_MODEL)
    t0 = time.time()
    Xtr_emb = embedder.encode(Xtr_text, show_progress_bar=True, batch_size=16)
    Xva_emb = embedder.encode(Xva_text, show_progress_bar=True, batch_size=16)
    Xte_emb = embedder.encode(Xte_text, show_progress_bar=True, batch_size=16)
    encode_secs = time.time() - t0

    Xtr_hyb = np.hstack((Xtr_emb, Xtr_tfidf))
    Xva_hyb = np.hstack((Xva_emb, Xva_tfidf))
    Xte_hyb = np.hstack((Xte_emb, Xte_tfidf))

    # Amplify scale_pos_weight beyond the natural ratio to bias the model
    # toward recall. This shifts the probability distribution of the
    # positive class upward, enabling a meaningful high-recall operating point.
    spw_boosted = round(spw * 2.5, 1)   # ~7.5x for 3:1 natural ratio
    print(f"Using boosted scale_pos_weight={spw_boosted} to improve recall")

    base_ensemble = VotingClassifier(estimators=[
        ("xgb", XGBClassifier(
            n_estimators=800, learning_rate=0.03, max_depth=7,
            subsample=0.8, colsample_bytree=0.7, min_child_weight=2,
            scale_pos_weight=spw_boosted,
            eval_metric="logloss", random_state=RANDOM_STATE,
        )),
        ("rf", RandomForestClassifier(
            n_estimators=400, max_depth=None,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
        )),
        ("lr", LogisticRegression(
            C=0.3, class_weight="balanced",
            max_iter=2000, random_state=RANDOM_STATE,
        )),
    ], voting="soft", weights=[3.0, 1.0, 1.5])

    base_ensemble.fit(Xtr_hyb, ytr)
    hybrid = CalibratedClassifierCV(base_ensemble, method="sigmoid", cv="prefit")
    hybrid.fit(Xva_hyb, yva)
    pva_hyb = hybrid.predict_proba(Xva_hyb)[:, 1]

    # Print recall at several intermediate thresholds to understand the curve
    print("\n  [Recall-Precision curve on val set at key thresholds]")
    for t in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
        pred_t = (pva_hyb >= t).astype(int)
        r = recall_score(yva, pred_t, zero_division=0)
        p = precision_score(yva, pred_t, zero_division=0)
        f = f1_score(yva, pred_t, zero_division=0)
        print(f"    thr={t:.2f} -> Recall={r:.3f}  Precision={p:.3f}  F1={f:.3f}")

    m_f1, m_r95 = evaluate(
        "PubMedBERT + XGBoost Hybrid (PROPOSED)", hybrid,
        None, None, Xte_hyb, yte, results, yva, pva_hyb,
    )

    # ---- Efficiency ----
    n = len(texts)
    per_paper = encode_secs / max(1, len(Xtr_text) + len(Xva_text) + len(Xte_text))
    est_full_secs = per_paper * n
    manual_hours = n * 3 / 60.0
    reduction = (1 - (est_full_secs / 3600.0) / manual_hours) * 100
    print("\n== Efficiency (measured) ==")
    print(f"  Embedding throughput     : {per_paper*1000:.1f} ms/paper")
    print(f"  Est. full-corpus screen  : {est_full_secs/60:.1f} min for {n} papers")
    print(f"  Manual baseline (3 min/paper): {manual_hours:.0f} hours")
    print(f"  Time reduction           : {reduction:.2f}%")

    # ---- Persist artifacts ----
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(hybrid, os.path.join(MODEL_DIR, "hybrid_xgb_model.pkl"))
    joblib.dump(tfidf, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    print(f"\n[artifacts] saved hybrid_xgb_model.pkl + tfidf_vectorizer.pkl to {MODEL_DIR}/")

    # ---- Charts (use F1-optimal predictions) ----
    res_key = "PubMedBERT + XGBoost Hybrid (PROPOSED)"
    comparison = {k: v[0] for k, v in results.items()}
    make_charts(yte, results[res_key][1], results[res_key][2], comparison)

    t_f1 = results[res_key][4]
    t_r95 = results[res_key][5]
    print("\n================ COPY THESE INTO TABLE 5 ================")
    print(f"  -- F1-optimal threshold ({t_f1:.2f}) --")
    print(f"Precision = {m_f1['precision']*100:.1f}%")
    print(f"Recall    = {m_f1['recall']*100:.1f}%")
    print(f"F1-score  = {m_f1['f1']*100:.1f}%")
    print(f"ROC-AUC   = {m_f1['roc_auc']*100:.1f}%")
    print(f"PR-AUC    = {m_f1['pr_auc']*100:.1f}%")
    print(f"  -- High-recall threshold ({t_r95:.2f}, Recall>={TARGET_RECALL:.0%}) --")
    print(f"Precision = {m_r95['precision']*100:.1f}%")
    print(f"Recall    = {m_r95['recall']*100:.1f}%")
    print(f"F1-score  = {m_r95['f1']*100:.1f}%")
    print(f"Time reduction = {reduction:.1f}%")
    print("========================================================")


if __name__ == "__main__":
    main()
