"""
CPU-OPTIMISED end-to-end fine-tuning of PubMedBERT for SR screening.
================================================================================
Same leakage-free methodology as finetune_pubmedbert.py (one stratified
70/15/15 split, class-weighted loss, thresholds tuned on validation only,
held-out test metrics + figures), but rewritten so it is *tractable on a CPU*.

CPU performance techniques applied here (each one is a real wall-clock win):

  1. DYNAMIC PADDING (DataCollatorWithPadding) instead of padding every row to
     max_length. On CPU, attention cost is ~O(seq_len^2); padding to the true
     batch maximum instead of 320 typically cuts compute 2-4x.
  2. LENGTH-GROUPED BATCHING (group_by_length): rows of similar length are
     batched together so padding waste is minimal.
  3. PARTIAL FREEZING (--freeze-layers N): the embedding table and the bottom N
     encoder layers are frozen. Only the top layers + classifier head get
     gradients, so backward() is far cheaper and you still adapt the
     decision-relevant layers. N=8 (of 12) is a good CPU default.
  4. THREAD PINNING (--threads): torch.set_num_threads to your physical core
     count. Over-subscribing hurts; 1 thread/core is best.
  5. SHORTER SEQUENCES (--max-len 256): abstracts rarely need 320 tokens; 256
     keeps almost all signal at lower quadratic cost.
  6. GRADIENT ACCUMULATION (--grad-accum): keep a small per-step batch (low RAM)
     while still getting a large effective batch for stable updates.
  7. OPTIONAL bf16 (--bf16): on modern CPUs (AVX-512/AMX) bf16 matmul is faster;
     auto-skipped if unsupported.
  8. SMALLER MODEL OPTION (--model): pass a distilled biomedical encoder for a
     3-5x speed-up at a small accuracy cost, e.g.
        --model nlpie/distil-biobert        (6-layer, ~half the FLOPs)
        --model nlpie/tiny-biobert          (fastest, lower ceiling)

Quick CPU recipe (good accuracy, runs in a fraction of the full-FT time):
    python experiments/baselines/finetune_pubmedbert_cpu.py \
        --epochs 3 --batch-size 8 --grad-accum 4 \
        --max-len 256 --freeze-layers 8 --threads <num_physical_cores>

Fastest recipe (for a quick chart refresh):
    python experiments/baselines/finetune_pubmedbert_cpu.py \
        --model nlpie/distil-biobert --epochs 2 --batch-size 16 \
        --max-len 224 --freeze-layers 3

Outputs (identical contract to the original script):
    sr_core/screening_model/finetuned_pubmedbert/   (HF model + tokenizer)
    sr_core/screening_model/finetuned_meta.json     (thresholds + metrics)
    experiments/results/*.png                        (leakage-free figures)
"""
import os
import sys
import json
import time
import argparse
import numpy as np

RANDOM_STATE = 42
DATA_PATH = os.path.join("data", "processed", "labeled_dataset.jsonl")
RESULTS_DIR = os.path.join("experiments", "results")
MODEL_DIR = os.path.join("sr_core", "screening_model")
SAVE_DIR = os.path.join(MODEL_DIR, "finetuned_pubmedbert")
META_PATH = os.path.join(MODEL_DIR, "finetuned_meta.json")

DEFAULT_MODEL = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
MAX_LEN = 256
TARGET_RECALL = 0.95


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="HF model id. Use a distilled biomedical model for speed.")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8, help="per-step batch (keep small on CPU)")
    p.add_argument("--grad-accum", type=int, default=4, help="gradient accumulation steps")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-len", type=int, default=MAX_LEN)
    p.add_argument("--freeze-layers", type=int, default=8,
                   help="freeze embeddings + bottom N encoder layers (0 = full FT)")
    p.add_argument("--threads", type=int, default=0,
                   help="torch CPU threads (0 = auto = physical cores)")
    p.add_argument("--bf16", action="store_true", help="use bf16 matmul if CPU supports it")
    p.add_argument("--target-recall", type=float, default=TARGET_RECALL)
    p.add_argument("--seed", type=int, default=RANDOM_STATE)
    p.add_argument("--init-seed", type=int, default=None,
                   help="seed for weight init / shuffling only; the train/val/test "
                        "split always uses --seed so every run shares one test set")
    p.add_argument("--limit", type=int, default=0, help="debug: cap dataset size (0 = all)")
    return p.parse_args()


def load_dataset(path, limit=0):
    texts, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = json.loads(line)
            title = (p.get("title") or "").strip()
            abstract = (p.get("abstract") or "").strip()
            texts.append(f"{title}. {abstract}".strip())
            labels.append(1 if p.get("human_label") == "include" else 0)
    if limit and limit < len(texts):
        texts, labels = texts[:limit], labels[:limit]
    return texts, np.array(labels)


def wss_at_recall(y_true, y_prob, target_recall):
    from sklearn.metrics import confusion_matrix
    n = len(y_true)
    best = (0.0, 0.5, 0.0)
    for t in np.unique(y_prob):
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
    # embeddings
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


def main():
    args = parse_args()

    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from torch.optim import AdamW
        from transformers import (
            AutoTokenizer, AutoModelForSequenceClassification,
            DataCollatorWithPadding, get_linear_schedule_with_warmup,
        )
    except Exception as e:
        sys.exit(f"Missing deps ({e}). Install: pip install torch transformers")

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        precision_score, recall_score, f1_score, accuracy_score,
        roc_auc_score, average_precision_score, confusion_matrix, classification_report,
    )

    if not os.path.exists(DATA_PATH):
        sys.exit(f"Dataset not found at {DATA_PATH}. Run from the project root.")

    # ---- CPU thread pinning ----
    n_threads = args.threads or (os.cpu_count() or 1)
    torch.set_num_threads(n_threads)
    try:
        torch.set_num_interop_threads(max(1, n_threads // 2))
    except Exception:
        pass
    os.environ.setdefault("OMP_NUM_THREADS", str(n_threads))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    init_seed = args.init_seed if args.init_seed is not None else args.seed
    np.random.seed(init_seed)
    torch.manual_seed(init_seed)
    device = "cpu"
    use_bf16 = args.bf16 and hasattr(torch, "bfloat16")
    print(f"Device: CPU | threads={n_threads} | bf16={use_bf16} | model={args.model}")

    texts, y = load_dataset(DATA_PATH, args.limit)
    print(f"Total: {len(y)} | Include: {int(y.sum())} | Exclude: {int((1 - y).sum())}")

    # ---- ONE stratified split: 70 / 15 / 15 (test is sacred) ----
    X_tmp, X_te, y_tmp, y_te = train_test_split(
        texts, y, test_size=0.15, stratify=y, random_state=args.seed)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tmp, y_tmp, test_size=0.1765, stratify=y_tmp, random_state=args.seed)
    print(f"Train: {len(X_tr)} | Val: {len(X_va)} | Test: {len(X_te)}")

    tok = AutoTokenizer.from_pretrained(args.model)

    def encode(batch):
        # NO padding here -> dynamic padding happens in the collator (CPU win)
        return tok(batch, truncation=True, max_length=args.max_len)

    enc_tr, enc_va, enc_te = encode(X_tr), encode(X_va), encode(X_te)

    class ScreenDataset(Dataset):
        def __init__(self, enc, labels):
            self.enc = enc
            self.labels = list(labels)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            item = {k: self.enc[k][i] for k in self.enc}
            item["labels"] = int(self.labels[i])
            return item

    ds_tr = ScreenDataset(enc_tr, y_tr)
    ds_va = ScreenDataset(enc_va, y_va)
    ds_te = ScreenDataset(enc_te, y_te)

    collator = DataCollatorWithPadding(tok)  # pads to per-batch max only

    # ---- length-grouped sampler for the train loader (minimise padding) ----
    lengths = [len(enc_tr["input_ids"][i]) for i in range(len(ds_tr))]

    def length_grouped_batches(lengths, bs, seed):
        rng = np.random.default_rng(seed)
        idx = np.argsort(lengths)
        # chunk into mega-batches, shuffle within, then form batches
        mega = bs * 50
        order = []
        for s in range(0, len(idx), mega):
            chunk = idx[s:s + mega].copy()
            rng.shuffle(chunk)
            order.extend(chunk.tolist())
        batches = [order[s:s + bs] for s in range(0, len(order), bs)]
        rng.shuffle(batches)
        return batches

    dl_va = DataLoader(ds_va, batch_size=max(8, args.batch_size), collate_fn=collator)
    dl_te = DataLoader(ds_te, batch_size=max(8, args.batch_size), collate_fn=collator)

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
    model.to(device)
    frozen = freeze_bottom_layers(model, args.freeze_layers)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Frozen params: {frozen:,} | Trainable: {trainable:,} "
          f"(freeze_layers={args.freeze_layers})")

    # ---- class-weighted loss (no SMOTE in embedding space) ----
    n_pos, n_neg = int(y_tr.sum()), int((1 - y_tr).sum())
    w_neg = (n_pos + n_neg) / (2.0 * n_neg)
    w_pos = (n_pos + n_neg) / (2.0 * n_pos)
    class_weights = torch.tensor([w_neg, w_pos], dtype=torch.float)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    print(f"Class weights [exclude, include]: [{w_neg:.3f}, {w_pos:.3f}]")

    optim = AdamW([p for p in model.parameters() if p.requires_grad],
                  lr=args.lr, weight_decay=0.01)
    steps_per_epoch = int(np.ceil(len(ds_tr) / args.batch_size / args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    sched = get_linear_schedule_with_warmup(
        optim, int(0.1 * total_steps), total_steps)

    def run_eval(dl):
        model.eval()
        probs, labels = [], []
        with torch.no_grad():
            for batch in dl:
                labels.extend(batch["labels"].numpy().tolist())
                inp = {k: v for k, v in batch.items() if k != "labels"}
                ctx = torch.autocast("cpu", dtype=torch.bfloat16) if use_bf16 \
                    else _nullcontext()
                with ctx:
                    logits = model(**inp).logits.float()
                p = torch.softmax(logits, dim=1)[:, 1].numpy()
                probs.extend(p.tolist())
        return np.array(labels), np.array(probs)

    t0 = time.time()
    best_val_auc, best_ep = -1.0, 0
    ckpt_dir = SAVE_DIR + "_ckpt"
    for ep in range(1, args.epochs + 1):
        model.train()
        batches = length_grouped_batches(lengths, args.batch_size, init_seed + ep)
        running, micro = 0.0, 0
        optim.zero_grad()
        for bi, batch_idx in enumerate(batches, 1):
            batch = collator([ds_tr[j] for j in batch_idx])
            labels = batch["labels"]
            inp = {k: v for k, v in batch.items() if k != "labels"}
            ctx = torch.autocast("cpu", dtype=torch.bfloat16) if use_bf16 else _nullcontext()
            with ctx:
                logits = model(**inp).logits.float()
                loss = loss_fn(logits, labels) / args.grad_accum
            loss.backward()
            running += loss.item() * args.grad_accum
            micro += 1
            if micro % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                optim.step()
                sched.step()
                optim.zero_grad()
            if bi % 50 == 0:
                print(f"  epoch {ep} batch {bi}/{len(batches)} "
                      f"loss {running/bi:.4f} elapsed {time.time()-t0:.0f}s")
        yv, pv = run_eval(dl_va)
        print(f"[epoch {ep}] train_loss={running/len(batches):.4f} "
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

    # ---- tune thresholds on VALIDATION only ----
    yv, pv = run_eval(dl_va)
    t_f1, best_f1 = f1_optimal_threshold(yv, pv)
    wss, t_recall, achieved_r = wss_at_recall(yv, pv, args.target_recall)
    print(f"\nVal thresholds -> F1-opt {t_f1:.2f} (F1={best_f1:.3f}) | "
          f"recall>={args.target_recall:.2f}: {t_recall:.2f} (WSS={wss:.3f})")

    # ---- held-out TEST (F1-optimal threshold) ----
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

    os.makedirs(SAVE_DIR, exist_ok=True)
    model.save_pretrained(SAVE_DIR)
    tok.save_pretrained(SAVE_DIR)
    meta = {
        "base_model": args.model, "max_len": args.max_len,
        "freeze_layers": args.freeze_layers, "best_epoch": best_ep,
        "init_seed": init_seed,
        "threshold_f1": t_f1, "threshold_recall95": t_recall,
        "target_recall": args.target_recall, "metrics_test": m,
        "wss_test": test_wss, "train_seconds": train_secs,
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"\n[artifacts] saved to {SAVE_DIR}/ and meta to {META_PATH}")

    make_charts(yt, pred, pt, m)

    print("\n================ COPY THESE INTO TABLE 5 ================")
    print(f"Precision = {m['precision']*100:.1f}%")
    print(f"Recall    = {m['recall']*100:.1f}%")
    print(f"F1-score  = {m['f1']*100:.1f}%")
    print(f"ROC-AUC   = {m['roc_auc']*100:.1f}%")
    print(f"PR-AUC    = {m['pr_auc']*100:.1f}%")
    print(f"WSS@{int(args.target_recall*100)}    = {test_wss*100:.1f}%")
    print(f"(train time: {train_secs/60:.1f} min on CPU)")
    print("========================================================")


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


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

    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_te, y_pred, labels=[0, 1])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Exclude", "Include"], yticklabels=["Exclude", "Include"])
    plt.title("Confusion Matrix - PubMedBERT fine-tuned (held-out test)")
    plt.ylabel("True Label"); plt.xlabel("Predicted Label"); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "transformer_confusion_matrix.png"), dpi=130)
    plt.close()

    fpr, tpr, _ = roc_curve(y_te, y_prob)
    plt.figure(figsize=(6.2, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2.2, label=f"ROC (AUC = {m['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="navy", lw=1.5)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - PubMedBERT fine-tuned (held-out, leakage-free)")
    plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "transformer_roc_curve.png"), dpi=130)
    plt.close()

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
