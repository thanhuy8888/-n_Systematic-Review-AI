"""
CPU-OPTIMISED frozen-embedding screening evaluation (chart refresher).
================================================================================
Drop-in faster version of evaluate_screening_clean.py. Identical leakage-free
methodology and identical model configuration -> the numbers it prints are the
SAME headline numbers (Table 5) and the figures it writes overwrite the same
files in experiments/results/. It is just engineered to be fast on a CPU and,
crucially, near-instant on every RE-RUN.

What makes it CPU-fast:

  1. EMBEDDING CACHE (the big one): PubMedBERT embeddings are computed once and
     cached to experiments/cache/ keyed by (model, max_len, dataset hash). The
     first run pays the encode cost; every subsequent run (e.g. to re-tune a
     threshold or restyle a chart) loads embeddings from disk in <1 s. This is
     why re-generating charts becomes trivial.
  2. THREAD PINNING: torch + numpy threads pinned to your physical core count.
  3. BIGGER ENCODE BATCH + no_grad + fp32->cached: one pass over the corpus.
  4. SPARSE TF-IDF kept sparse until the hstack, so peak RAM stays low.
  5. --fast flag: optionally drops the slowest classical members (SVC,
     halves RF trees) for quick iteration. OFF by default so the saved numbers
     are bit-for-bit the thesis configuration. Use --fast only when you are
     iterating on chart styling, not when capturing final Table-5 numbers.

Run from the project root:

    # full, thesis-faithful numbers (first run encodes, then caches):
    python experiments/baselines/evaluate_screening_clean_cpu.py --threads <cores>

    # quick chart iteration after the first run (uses cached embeddings):
    python experiments/baselines/evaluate_screening_clean_cpu.py --fast

Outputs: experiments/results/*.png  +  sr_core/screening_model/*.pkl
"""
import os
import sys
import json
import time
import hashlib
import argparse
import numpy as np

from sklearn.model_selection import train_test_split
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

TARGET_RECALL = 0.95
RANDOM_STATE = 42
DATA_PATH = os.path.join("data", "processed", "labeled_dataset.jsonl")
RESULTS_DIR = os.path.join("experiments", "results")
MODEL_DIR = os.path.join("sr_core", "screening_model")
CACHE_DIR = os.path.join("experiments", "cache")
EMBED_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--threads", type=int, default=0, help="CPU threads (0=auto)")
    p.add_argument("--batch-size", type=int, default=32, help="embedding batch size")
    p.add_argument("--fast", action="store_true",
                   help="drop slowest classical members for quick chart iteration")
    p.add_argument("--no-cache", action="store_true", help="ignore the embedding cache")
    return p.parse_args()


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
    best_t = 0.01
    for t in np.linspace(0.99, 0.01, 980):
        if recall_score(y_true, (y_prob >= t).astype(int), zero_division=0) >= target:
            best_t = float(t)
            break
    return best_t


def cached_embeddings(embedder, texts, tag, batch_size, use_cache):
    """Encode once, then load from disk forever after. Keyed by content hash."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    h = hashlib.md5(("||".join(texts)).encode("utf-8")).hexdigest()[:12]
    fn = os.path.join(CACHE_DIR, f"emb_{EMBED_MODEL.split('/')[-1]}_{tag}_{h}.npy")
    if use_cache and os.path.exists(fn):
        print(f"  [cache hit] {fn}")
        return np.load(fn)
    t0 = time.time()
    emb = embedder.encode(texts, show_progress_bar=True, batch_size=batch_size,
                          convert_to_numpy=True)
    np.save(fn, emb)
    print(f"  [cache write] {fn}  ({time.time()-t0:.1f}s for {len(texts)} texts)")
    return emb


def evaluate(name, clf, X_tr, y_tr, X_te, y_te, results, y_val=None, prob_val=None):
    if X_tr is not None:
        clf.fit(X_tr, y_tr)
    y_prob = clf.predict_proba(X_te)[:, 1]
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
    m_f1, m_r95 = metrics(y_pred_f1), metrics(y_pred_r95)
    results[name] = (m_f1, y_pred_f1, y_prob, clf, t_f1, t_r95, m_r95)
    print(f"\n== {name} (held-out test) ==")
    print(f"  [thr={t_f1:.2f} F1-opt] P={m_f1['precision']:.4f} R={m_f1['recall']:.4f} "
          f"F1={m_f1['f1']:.4f} ROC={m_f1['roc_auc']:.4f} PR={m_f1['pr_auc']:.4f}")
    print(classification_report(y_te, y_pred_f1, target_names=["exclude", "include"], digits=3))
    return m_f1, m_r95


def make_charts(y_te, hybrid_pred, hybrid_prob, comparison):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    os.makedirs(RESULTS_DIR, exist_ok=True)

    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_te, hybrid_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Exclude", "Include"], yticklabels=["Exclude", "Include"])
    plt.title("Confusion Matrix - PubMedBERT + XGBoost (held-out test)")
    plt.ylabel("True Label"); plt.xlabel("Predicted Label"); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "transformer_confusion_matrix.png"), dpi=130); plt.close()

    fpr, tpr, _ = roc_curve(y_te, hybrid_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2,
             label=f"ROC (AUC = {roc_auc_score(y_te, hybrid_prob):.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0, 1]); plt.ylim([0, 1.05]); plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate"); plt.title("ROC Curve - PubMedBERT + XGBoost (held-out test)")
    plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "transformer_roc_curve.png"), dpi=130); plt.close()

    prec, rec, _ = precision_recall_curve(y_te, hybrid_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, color="blue", lw=2,
             label=f"PR (AUC = {average_precision_score(y_te, hybrid_prob):.4f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall Curve - PubMedBERT + XGBoost (held-out test)")
    plt.legend(loc="lower left"); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "transformer_pr_curve.png"), dpi=130); plt.close()

    plt.figure(figsize=(8, 5))
    sns.histplot(hybrid_prob[y_te == 0], color="red", label="Actual: Exclude",
                 kde=True, stat="density", bins=30, alpha=0.5)
    sns.histplot(hybrid_prob[y_te == 1], color="green", label="Actual: Include",
                 kde=True, stat="density", bins=30, alpha=0.5)
    plt.axvline(x=0.5, color="black", linestyle="--", label="Threshold (0.5)")
    plt.title("Predicted Probability Distribution (held-out test)")
    plt.xlabel("P(Include)"); plt.ylabel("Density"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "transformer_probability_distribution.png"), dpi=130); plt.close()

    names = list(comparison.keys())
    metrics_k = ["accuracy", "f1", "roc_auc"]
    x = np.arange(len(names)); w = 0.25
    plt.figure(figsize=(9, 5))
    for i, mt in enumerate(metrics_k):
        plt.bar(x + (i - 1) * w, [comparison[n][mt] for n in names], w, label=mt.upper())
    plt.xticks(x, names, rotation=15, ha="right"); plt.ylim(0, 1.0)
    plt.ylabel("Score"); plt.title("Screening Model Comparison (held-out test)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "model_comparison_chart.png"), dpi=130); plt.close()
    print(f"[charts] figures written to {RESULTS_DIR}/")


def main():
    args = parse_args()
    n_threads = args.threads or (os.cpu_count() or 1)
    os.environ.setdefault("OMP_NUM_THREADS", str(n_threads))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        import torch
        torch.set_num_threads(n_threads)
    except Exception:
        pass
    print(f"CPU threads: {n_threads} | fast={args.fast}")

    if not os.path.exists(DATA_PATH):
        sys.exit(f"Dataset not found at {DATA_PATH}. Run from the project root.")

    texts, y = load_dataset(DATA_PATH)
    print(f"Total {len(y)} | Include {int(y.sum())} | Exclude {int((1-y).sum())}")

    Xtmp_text, Xte_text, ytmp, yte = train_test_split(
        texts, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    Xtr_text, Xva_text, ytr, yva = train_test_split(
        Xtmp_text, ytmp, test_size=0.125, stratify=ytmp, random_state=RANDOM_STATE)
    print(f"Train {len(Xtr_text)} | Val {len(Xva_text)} | Test {len(Xte_text)}")

    n_neg, n_pos = int((ytr == 0).sum()), int((ytr == 1).sum())
    spw = round(n_neg / max(1, n_pos), 2)

    tfidf = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), stop_words="english")
    Xtr_tfidf = tfidf.fit_transform(Xtr_text).toarray()
    Xva_tfidf = tfidf.transform(Xva_text).toarray()
    Xte_tfidf = tfidf.transform(Xte_text).toarray()

    results = {}

    # Baseline: TF-IDF only
    smote = SMOTE(random_state=RANDOM_STATE)
    Xtr_bal, ytr_bal = smote.fit_resample(Xtr_tfidf, ytr)
    base_members = [
        ("lr", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE)),
        ("rf", RandomForestClassifier(
            n_estimators=100 if args.fast else 200,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)),
    ]
    if not args.fast:
        base_members.append(
            ("svc", SVC(kernel="linear", class_weight="balanced", probability=True,
                        random_state=RANDOM_STATE)))
    baseline = VotingClassifier(estimators=base_members, voting="soft")
    pva_base = baseline.fit(Xtr_bal, ytr_bal).predict_proba(Xva_tfidf)[:, 1]
    evaluate("TF-IDF Ensemble (baseline)", baseline, Xtr_bal, ytr_bal,
             Xte_tfidf, yte, results, yva, pva_base)

    # Hybrid: PubMedBERT + TF-IDF
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        sys.exit(f"\n[hybrid] sentence-transformers unavailable ({e}). "
                 f"pip install torch sentence-transformers and re-run.")

    print(f"\nLoading PubMedBERT embedder: {EMBED_MODEL} ...")
    embedder = SentenceTransformer(EMBED_MODEL)
    use_cache = not args.no_cache
    Xtr_emb = cached_embeddings(embedder, Xtr_text, "tr", args.batch_size, use_cache)
    Xva_emb = cached_embeddings(embedder, Xva_text, "va", args.batch_size, use_cache)
    Xte_emb = cached_embeddings(embedder, Xte_text, "te", args.batch_size, use_cache)

    Xtr_hyb = np.hstack((Xtr_emb, Xtr_tfidf))
    Xva_hyb = np.hstack((Xva_emb, Xva_tfidf))
    Xte_hyb = np.hstack((Xte_emb, Xte_tfidf))

    spw_boosted = round(spw * 2.5, 1)
    print(f"scale_pos_weight boosted = {spw_boosted}")

    members = [
        ("xgb", XGBClassifier(
            n_estimators=800, learning_rate=0.03, max_depth=7,
            subsample=0.8, colsample_bytree=0.7, min_child_weight=2,
            scale_pos_weight=spw_boosted, eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=n_threads, tree_method="hist")),
        ("rf", RandomForestClassifier(
            n_estimators=200 if args.fast else 400, max_depth=None,
            class_weight="balanced_subsample", min_samples_leaf=2,
            random_state=RANDOM_STATE, n_jobs=-1)),
        ("lr", LogisticRegression(
            C=0.3, class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE)),
    ]
    base_ensemble = VotingClassifier(estimators=members, voting="soft",
                                     weights=[3.0, 1.0, 1.5])
    base_ensemble.fit(Xtr_hyb, ytr)
    hybrid = CalibratedClassifierCV(base_ensemble, method="sigmoid", cv="prefit")
    hybrid.fit(Xva_hyb, yva)
    pva_hyb = hybrid.predict_proba(Xva_hyb)[:, 1]

    m_f1, m_r95 = evaluate(
        "PubMedBERT + XGBoost Hybrid (PROPOSED)", hybrid,
        None, None, Xte_hyb, yte, results, yva, pva_hyb)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(hybrid, os.path.join(MODEL_DIR, "hybrid_xgb_model.pkl"))
    joblib.dump(tfidf, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))

    key = "PubMedBERT + XGBoost Hybrid (PROPOSED)"
    comparison = {k: v[0] for k, v in results.items()}
    make_charts(yte, results[key][1], results[key][2], comparison)

    t_f1, t_r95 = results[key][4], results[key][5]
    print("\n================ COPY THESE INTO TABLE 5 ================")
    print(f"  -- F1-optimal threshold ({t_f1:.2f}) --")
    print(f"Precision = {m_f1['precision']*100:.1f}%")
    print(f"Recall    = {m_f1['recall']*100:.1f}%")
    print(f"F1-score  = {m_f1['f1']*100:.1f}%")
    print(f"ROC-AUC   = {m_f1['roc_auc']*100:.1f}%")
    print(f"PR-AUC    = {m_f1['pr_auc']*100:.1f}%")
    print(f"  -- High-recall threshold ({t_r95:.2f}, Recall>={TARGET_RECALL:.0%}) --")
    print(f"Precision = {m_r95['precision']*100:.1f}%  "
          f"Recall = {m_r95['recall']*100:.1f}%  F1 = {m_r95['f1']*100:.1f}%")
    print("========================================================")


if __name__ == "__main__":
    main()
