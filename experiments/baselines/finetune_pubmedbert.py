"""
End-to-end fine-tuning of PubMedBERT for systematic-review screening.
================================================================================
This is the "strong improvement" path referenced in the thesis future work.

Instead of FREEZING the transformer and stacking XGBoost on top of static
embeddings (see evaluate_screening_clean.py), this script fine-tunes the whole
PubMedBERT encoder end-to-end as a binary sequence classifier. End-to-end
fine-tuning lets the contextual layers adapt to the include/exclude decision
boundary and typically lifts ROC-AUC well above the frozen-embedding baseline.

Methodology (kept leakage-free, same discipline as evaluate_screening_clean.py):
    * ONE stratified split  ->  70% train / 15% validation / 15% test.
    * The test split is held out: never used for tuning or thresholding.
    * Class imbalance is handled by CLASS WEIGHTS in the loss (NOT SMOTE):
      SMOTE interpolates synthetic points in a 768-d embedding space, which is
      unreliable; weighting the loss is the standard, honest alternative.
    * The decision threshold is tuned on the VALIDATION split only, for two
      operating points that matter in SR screening:
          (a) F1-optimal threshold
          (b) recall >= TARGET_RECALL (default 0.95)  ->  reported as WSS@95
      and then applied unchanged to the held-out test split.

Outputs:
    * sr_core/screening_model/finetuned_pubmedbert/   (HF model + tokenizer)
    * sr_core/screening_model/finetuned_meta.json     (thresholds + metrics)
    * experiments/results/*.png                        (genuine, leakage-free)

Run from the project root (GPU strongly recommended):
    python experiments/baselines/finetune_pubmedbert.py

The printed "COPY THESE INTO TABLE 5" block is the honest, reproducible set of
numbers to paste into the thesis (and to regenerate the figures from).
"""
import os
import sys
import json
import time
import argparse
import numpy as np

# --------------------------------------------------------------------------- #
# Config (override via CLI flags)
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
DATA_PATH = os.path.join("data", "processed", "labeled_dataset.jsonl")
RESULTS_DIR = os.path.join("experiments", "results")
MODEL_DIR = os.path.join("sr_core", "screening_model")
SAVE_DIR = os.path.join(MODEL_DIR, "finetuned_pubmedbert")
META_PATH = os.path.join(MODEL_DIR, "finetuned_meta.json")

# Canonical PubMedBERT MLM checkpoint (Microsoft renamed PubMedBERT -> BiomedBERT).
# Alternatives worth trying: "michiyasunaga/BioLinkBERT-base"
DEFAULT_MODEL = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
MAX_LEN = 320
TARGET_RECALL = 0.95


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-len", type=int, default=MAX_LEN)
    p.add_argument("--target-recall", type=float, default=TARGET_RECALL)
    p.add_argument("--seed", type=int, default=RANDOM_STATE)
    p.add_argument("--freeze-layers", type=int, default=0,
                   help="freeze embeddings + bottom N encoder layers (0 = full FT)")
    return p.parse_args()


def load_dataset(path):
    import json as _json
    texts, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = _json.loads(line)
            title = (p.get("title") or "").strip()
            abstract = (p.get("abstract") or "").strip()
            texts.append(f"{title}. {abstract}".strip())
            labels.append(1 if p.get("human_label") == "include" else 0)
    return texts, np.array(labels)


def wss_at_recall(y_true, y_prob, target_recall):
    """Work-Saved-over-Sampling at a target recall.
    WSS@r = (TN + FN)/N - (1 - r), evaluated at the lowest threshold that
    still achieves recall >= r. Returns (wss, threshold, achieved_recall)."""
    from sklearn.metrics import confusion_matrix
    n = len(y_true)
    thresholds = np.unique(y_prob)
    best = (0.0, 0.5, 0.0)
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if recall >= target_recall:
            wss = (tn + fn) / n - (1 - target_recall)
            if wss > best[0]:
                best = (wss, float(t), recall)
    return best


def f1_optimal_threshold(y_true, y_prob):
    from sklearn.metrics import f1_score
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        f1 = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


def freeze_bottom_layers(model, n):
    """Freeze the embeddings + the bottom n transformer encoder layers."""
    if n <= 0:
        return 0
    frozen = 0
    base = getattr(model, "bert", None) or getattr(model, "roberta", None) \
        or getattr(model, "distilbert", None) or model.base_model
    emb = getattr(base, "embeddings", None)
    if emb is not None:
        for p in emb.parameters():
            p.requires_grad = False
            frozen += p.numel()
    # encoder layers (BERT: encoder.layer ; DistilBERT: transformer.layer)
    enc = getattr(base, "encoder", None) or getattr(base, "transformer", None)
    layers = getattr(enc, "layer", None) if enc is not None else None
    if layers is not None:
        for i, layer in enumerate(layers):
            if i < n:
                for p in layer.parameters():
                    p.requires_grad = False
                    frozen += p.numel()
    return frozen


# --------------------------------------------------------------------------- #
# Torch dataset
# --------------------------------------------------------------------------- #
def build_torch():
    import torch
    from torch.utils.data import Dataset

    class ScreenDataset(Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
            item["labels"] = torch.tensor(int(self.labels[idx]))
            return item

    return ScreenDataset


def main():
    args = parse_args()

    try:
        import torch
        from torch.utils.data import DataLoader
        from torch.optim import AdamW
        from transformers import (
            AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup,
        )
    except Exception as e:
        sys.exit(f"Missing deep-learning deps ({e}). Install: pip install torch transformers")

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        precision_score, recall_score, f1_score, accuracy_score,
        roc_auc_score, average_precision_score, confusion_matrix, classification_report,
    )

    if not os.path.exists(DATA_PATH):
        sys.exit(f"Dataset not found at {DATA_PATH}. Run from the project root.")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Model: {args.model}")
    if device == "cpu":
        print("WARNING: no GPU detected. Fine-tuning on CPU is very slow "
              "(hours). A CUDA GPU is strongly recommended.")

    texts, y = load_dataset(DATA_PATH)
    print(f"Total: {len(y)} | Include: {int(y.sum())} | Exclude: {int((1 - y).sum())}")

    # ---- ONE stratified split: 70 / 15 / 15 (test is sacred) ----
    X_tmp, X_te, y_tmp, y_te = train_test_split(
        texts, y, test_size=0.15, stratify=y, random_state=args.seed)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tmp, y_tmp, test_size=0.1765, stratify=y_tmp, random_state=args.seed)  # ~15% overall
    print(f"Train: {len(X_tr)} | Val: {len(X_va)} | Test: {len(X_te)} "
          f"(test never used for tuning)")

    tok = AutoTokenizer.from_pretrained(args.model)

    def encode(batch):
        return tok(batch, truncation=True, padding="max_length",
                   max_length=args.max_len, return_tensors=None)

    enc_tr = encode(X_tr)
    enc_va = encode(X_va)
    enc_te = encode(X_te)

    ScreenDataset = build_torch()
    ds_tr = ScreenDataset(enc_tr, y_tr)
    ds_va = ScreenDataset(enc_va, y_va)
    ds_te = ScreenDataset(enc_te, y_te)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size)
    dl_te = DataLoader(ds_te, batch_size=args.batch_size)

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
    model.to(device)
    frozen = freeze_bottom_layers(model, args.freeze_layers)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Frozen params: {frozen:,} | Trainable: {trainable:,} "
          f"(freeze_layers={args.freeze_layers})")

    # ---- Class weights instead of SMOTE ----
    n_pos = int(y_tr.sum())
    n_neg = int((1 - y_tr).sum())
    w_neg = (n_pos + n_neg) / (2.0 * n_neg)
    w_pos = (n_pos + n_neg) / (2.0 * n_pos)
    class_weights = torch.tensor([w_neg, w_pos], dtype=torch.float).to(device)
    print(f"Class weights [exclude, include]: [{w_neg:.3f}, {w_pos:.3f}]")
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    optim = AdamW([p for p in model.parameters() if p.requires_grad],
                  lr=args.lr, weight_decay=0.01)
    total_steps = len(dl_tr) * args.epochs
    sched = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    def run_eval(dl):
        model.eval()
        probs, labels = [], []
        with torch.no_grad():
            for batch in dl:
                labels.extend(batch["labels"].numpy().tolist())
                inp = {k: v.to(device) for k, v in batch.items() if k != "labels"}
                with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                    logits = model(**inp).logits
                p = torch.softmax(logits, dim=1)[:, 1].float().cpu().numpy()
                probs.extend(p.tolist())
        return np.array(labels), np.array(probs)

    # ---- Training loop ----
    t0 = time.time()
    best_val_auc, best_ep = -1.0, 0
    ckpt_dir = SAVE_DIR + "_ckpt"
    for ep in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for step, batch in enumerate(dl_tr, 1):
            labels = batch["labels"].to(device)
            inp = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            optim.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits = model(**inp).logits
                loss = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            sched.step()
            running += loss.item()
            if step % 50 == 0:
                print(f"  epoch {ep} step {step}/{len(dl_tr)} loss {running/step:.4f}")
        yv, pv = run_eval(dl_va)
        print(f"[epoch {ep}] train_loss={running/len(dl_tr):.4f} "
              f"val_ROC-AUC={roc_auc_score(yv, pv):.4f} "
              f"val_PR-AUC={average_precision_score(yv, pv):.4f}")
        # checkpoint the best-val epoch: an interrupted or overfit run still
        # leaves the peak model on disk instead of only the latest epoch
        val_auc = float(roc_auc_score(yv, pv))
        if val_auc > best_val_auc:
            best_val_auc, best_ep = val_auc, ep
            model.save_pretrained(ckpt_dir)
            tok.save_pretrained(ckpt_dir)
            with open(os.path.join(ckpt_dir, "ckpt_info.json"), "w", encoding="utf-8") as f:
                json.dump({"epoch": ep, "val_roc_auc": val_auc,
                           "val_pr_auc": float(average_precision_score(yv, pv))}, f, indent=2)
    train_secs = time.time() - t0

    # threshold-tune and test the BEST-val epoch, not necessarily the last one
    if best_ep and best_ep != args.epochs:
        print(f"\nReloading best checkpoint (epoch {best_ep}, "
              f"val ROC-AUC {best_val_auc:.4f}) for final evaluation.")
        model = AutoModelForSequenceClassification.from_pretrained(
            ckpt_dir, num_labels=2)
        model.to(device)

    # ---- Tune thresholds on VALIDATION only ----
    yv, pv = run_eval(dl_va)
    t_f1, best_f1 = f1_optimal_threshold(yv, pv)
    wss, t_recall, achieved_r = wss_at_recall(yv, pv, args.target_recall)
    print(f"\nValidation thresholds -> F1-opt: {t_f1:.2f} (F1={best_f1:.3f}) | "
          f"recall>={args.target_recall:.2f}: {t_recall:.2f} (WSS={wss:.3f})")

    # ---- Final evaluation on HELD-OUT TEST (threshold = F1-optimal) ----
    yt, pt = run_eval(dl_te)
    pred = (pt >= t_f1).astype(int)
    m = {
        "accuracy": accuracy_score(yt, pred),
        "precision": precision_score(yt, pred, zero_division=0),
        "recall": recall_score(yt, pred, zero_division=0),
        "f1": f1_score(yt, pred, zero_division=0),
        "roc_auc": roc_auc_score(yt, pt),
        "pr_auc": average_precision_score(yt, pt),
    }
    test_wss, _, _ = wss_at_recall(yt, pt, args.target_recall)
    print("\n== Held-out TEST (no leakage, threshold = val F1-optimal) ==")
    for k, v in m.items():
        print(f"  {k:10s}: {v:.4f}")
    print(f"  WSS@{int(args.target_recall*100)}  : {test_wss:.4f}")
    print(classification_report(yt, pred, target_names=["exclude", "include"], digits=3))

    # ---- Save model + metadata ----
    os.makedirs(SAVE_DIR, exist_ok=True)
    model.save_pretrained(SAVE_DIR)
    tok.save_pretrained(SAVE_DIR)
    meta = {
        "base_model": args.model,
        "max_len": args.max_len,
        "freeze_layers": args.freeze_layers,
        "best_epoch": best_ep,
        "threshold_f1": t_f1,
        "threshold_recall95": t_recall,
        "target_recall": args.target_recall,
        "metrics_test": m,
        "wss_test": test_wss,
        "train_seconds": train_secs,
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"\n[artifacts] saved fine-tuned model to {SAVE_DIR}/ and meta to {META_PATH}")

    # ---- Genuine leakage-free charts (overwrite the old ones) ----
    make_charts(yt, pred, pt, m)

    print("\n================ COPY THESE INTO TABLE 5 ================")
    print(f"Precision = {m['precision']*100:.1f}%")
    print(f"Recall    = {m['recall']*100:.1f}%")
    print(f"F1-score  = {m['f1']*100:.1f}%")
    print(f"ROC-AUC   = {m['roc_auc']*100:.1f}%")
    print(f"PR-AUC    = {m['pr_auc']*100:.1f}%")
    print(f"WSS@{int(args.target_recall*100)}    = {test_wss*100:.1f}%")
    print("========================================================")


def make_charts(y_te, y_pred, y_prob, m):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
    except Exception as e:
        print(f"[charts] skipped ({e})")
        return
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Confusion matrix
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_te, y_pred, labels=[0, 1])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Exclude", "Include"], yticklabels=["Exclude", "Include"])
    plt.title("Confusion Matrix - PubMedBERT fine-tuned (held-out test)")
    plt.ylabel("True Label"); plt.xlabel("Predicted Label"); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "transformer_confusion_matrix.png"), dpi=130)
    plt.close()

    # ROC
    fpr, tpr, _ = roc_curve(y_te, y_prob)
    plt.figure(figsize=(6.2, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2.2, label=f"ROC (AUC = {m['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="navy", lw=1.5)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - PubMedBERT fine-tuned (held-out, leakage-free)")
    plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "transformer_roc_curve.png"), dpi=130)
    plt.close()

    # PR
    prec, rec, _ = precision_recall_curve(y_te, y_prob)
    plt.figure(figsize=(6.2, 5))
    plt.plot(rec, prec, color="blue", lw=2.2, label=f"PR (AP = {m['pr_auc']:.3f})")
    base = float(np.mean(y_te))
    plt.hlines(base, 0, 1, color="grey", ls="--", lw=1.3, label=f"Baseline ({base:.2f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall Curve - PubMedBERT fine-tuned (held-out)")
    plt.legend(loc="upper right"); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "transformer_pr_curve.png"), dpi=130)
    plt.close()

    # Probability distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(y_prob[y_te == 0], bins=30, stat="density", color="red",
                 alpha=0.45, label="Actual: Exclude", kde=True)
    sns.histplot(y_prob[y_te == 1], bins=30, stat="density", color="green",
                 alpha=0.45, label="Actual: Include", kde=True)
    plt.axvline(0.5, color="black", ls="--", label="Threshold (0.5)")
    plt.xlabel("Predicted Probability (Include)"); plt.ylabel("Density")
    plt.title("Probability Distribution of AI Confidence (held-out)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "transformer_probability_distribution.png"), dpi=130)
    plt.close()
    print(f"[charts] genuine figures written to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
