"""
CPU 5-seed soft-voting ENSEMBLE trainer for SR screening.
================================================================================
Trains N independently-seeded fine-tuned distil-BioBERT models that all share a
single leakage-free 70/15/15 split (the test set is identical across seeds, so
their probabilities can be averaged), then:

  * saves every seed model to  sr_core/screening_model/ensemble/seed_<s>/
  * averages the per-abstract include-probabilities (soft voting)
  * tunes F1-optimal / F2-optimal (recall-oriented) / recall@95 thresholds on the
    ENSEMBLE validation probabilities only
  * reports held-out TEST metrics (ROC-AUC, PR-AUC, WSS@95, ...)
  * writes  sr_core/screening_model/ensemble/ensemble_meta.json
            (deploy_threshold = threshold_f2 = recall-oriented operating point)

The single fine-tuned model in finetuned_pubmedbert/ is left untouched so it
remains a fallback; transformer_screen.py auto-prefers the ensemble when present.

Run from the project root:
    venv/Scripts/python.exe experiments/baselines/train_ensemble_cpu.py \
        --seeds 42 123 2026 777 1234 --epochs 6 --freeze-layers 2 --max-len 256
"""
import os
import sys
import json
import time
import argparse
import numpy as np

RANDOM_STATE = 42
DATA_PATH = os.path.join("data", "processed", "labeled_dataset.jsonl")
MODEL_DIR = os.path.join("sr_core", "screening_model")
ENS_DIR = os.path.join(MODEL_DIR, "ensemble")
PROBS_DIR = os.path.join(ENS_DIR, "probs")
DEFAULT_MODEL = "nlpie/distil-biobert"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2026, 777, 1234])
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--freeze-layers", type=int, default=2)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--split-seed", type=int, default=RANDOM_STATE,
                   help="seed for the train/val/test split (shared by every model)")
    p.add_argument("--target-recall", type=float, default=0.95)
    p.add_argument("--limit", type=int, default=0,
                   help="debug only: cap dataset size for a fast smoke test (0 = all)")
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
        rng = np.random.default_rng(0)
        idx = rng.choice(len(texts), size=limit, replace=False)
        texts = [texts[i] for i in idx]
        labels = [labels[i] for i in idx]
    return texts, np.array(labels)


def freeze_bottom_layers(model, n):
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
    enc = getattr(base, "encoder", None) or getattr(base, "transformer", None)
    layers = getattr(enc, "layer", None) if enc is not None else None
    if layers is not None:
        for i, layer in enumerate(layers):
            if i < n:
                for p in layer.parameters():
                    p.requires_grad = False
                    frozen += p.numel()
    return frozen


def wss_at_recall(y_true, y_prob, target):
    from sklearn.metrics import confusion_matrix
    n = len(y_true)
    best = (0.0, 0.5, 0.0)
    for t in np.unique(y_prob):
        pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if recall >= target:
            wss = (tn + fn) / n - (1 - target)
            if wss > best[0]:
                best = (wss, float(t), recall)
    return best


def best_threshold(y_true, y_prob, beta):
    from sklearn.metrics import fbeta_score
    bt, bs = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 181):
        s = fbeta_score(y_true, (y_prob >= t).astype(int), beta=beta, zero_division=0)
        if s > bs:
            bs, bt = s, float(t)
    return bt, bs


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def main():
    args = parse_args()
    import torch
    from torch.utils.data import Dataset
    from torch.optim import AdamW
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        DataCollatorWithPadding, get_linear_schedule_with_warmup,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        precision_score, recall_score, f1_score, accuracy_score,
        roc_auc_score, average_precision_score,
    )

    if not os.path.exists(DATA_PATH):
        sys.exit(f"Dataset not found at {DATA_PATH}. Run from project root.")

    n_threads = args.threads or (os.cpu_count() or 1)
    torch.set_num_threads(n_threads)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    texts, y = load_dataset(DATA_PATH, args.limit)
    X_tmp, X_te, y_tmp, y_te = train_test_split(
        texts, y, test_size=0.15, stratify=y, random_state=args.split_seed)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tmp, y_tmp, test_size=0.1765, stratify=y_tmp, random_state=args.split_seed)
    print(f"Total {len(y)} | Train {len(X_tr)} | Val {len(X_va)} | Test {len(X_te)} | "
          f"threads={n_threads} | model={args.model}")

    tok = AutoTokenizer.from_pretrained(args.model)

    def encode(batch):
        return tok(batch, truncation=True, max_length=args.max_len)

    enc_tr, enc_va, enc_te = encode(X_tr), encode(X_va), encode(X_te)

    class DS(Dataset):
        def __init__(self, enc, labels):
            self.enc, self.labels = enc, list(labels)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            item = {k: self.enc[k][i] for k in self.enc}
            item["labels"] = int(self.labels[i])
            return item

    ds_tr, ds_va, ds_te = DS(enc_tr, y_tr), DS(enc_va, y_va), DS(enc_te, y_te)
    collator = DataCollatorWithPadding(tok)
    lengths = [len(enc_tr["input_ids"][i]) for i in range(len(ds_tr))]

    def length_grouped_batches(bs, seed):
        rng = np.random.default_rng(seed)
        idx = np.argsort(lengths)
        mega = bs * 50
        order = []
        for s in range(0, len(idx), mega):
            chunk = idx[s:s + mega].copy()
            rng.shuffle(chunk)
            order.extend(chunk.tolist())
        batches = [order[s:s + bs] for s in range(0, len(order), bs)]
        rng.shuffle(batches)
        return batches

    from torch.utils.data import DataLoader
    dl_va = DataLoader(ds_va, batch_size=max(8, args.batch_size), collate_fn=collator)
    dl_te = DataLoader(ds_te, batch_size=max(8, args.batch_size), collate_fn=collator)

    n_pos, n_neg = int(y_tr.sum()), int((1 - y_tr).sum())
    class_weights = torch.tensor(
        [(n_pos + n_neg) / (2.0 * n_neg), (n_pos + n_neg) / (2.0 * n_pos)], dtype=torch.float)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    def run_eval(model, dl):
        model.eval()
        probs, labels = [], []
        with torch.no_grad():
            for batch in dl:
                labels.extend(batch["labels"].numpy().tolist())
                inp = {k: v for k, v in batch.items() if k != "labels"}
                logits = model(**inp).logits.float()
                probs.extend(torch.softmax(logits, dim=1)[:, 1].numpy().tolist())
        return np.array(labels), np.array(probs)

    def train_one(init_seed, out_dir):
        np.random.seed(init_seed)
        torch.manual_seed(init_seed)
        model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
        freeze_bottom_layers(model, args.freeze_layers)
        optim = AdamW([p for p in model.parameters() if p.requires_grad],
                      lr=args.lr, weight_decay=0.01)
        steps = int(np.ceil(len(ds_tr) / args.batch_size / args.grad_accum)) * args.epochs
        sched = get_linear_schedule_with_warmup(optim, int(0.1 * steps), steps)
        ckpt = out_dir + "_ckpt"
        best_auc, best_ep = -1.0, 0
        t0 = time.time()
        for ep in range(1, args.epochs + 1):
            model.train()
            batches = length_grouped_batches(args.batch_size, init_seed + ep)
            optim.zero_grad()
            micro = 0
            for batch_idx in batches:
                batch = collator([ds_tr[j] for j in batch_idx])
                labels = batch["labels"]
                inp = {k: v for k, v in batch.items() if k != "labels"}
                logits = model(**inp).logits.float()
                loss = loss_fn(logits, labels) / args.grad_accum
                loss.backward()
                micro += 1
                if micro % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], 1.0)
                    optim.step()
                    sched.step()
                    optim.zero_grad()
            yv, pv = run_eval(model, dl_va)
            auc = float(roc_auc_score(yv, pv))
            print(f"    seed {init_seed} epoch {ep}/{args.epochs} "
                  f"val_ROC-AUC={auc:.4f} elapsed={time.time()-t0:.0f}s", flush=True)
            if auc > best_auc:
                best_auc, best_ep = auc, ep
                model.save_pretrained(ckpt)
                tok.save_pretrained(ckpt)
        # reload best-val epoch
        if best_ep != args.epochs:
            model = AutoModelForSequenceClassification.from_pretrained(ckpt, num_labels=2)
        model.save_pretrained(out_dir)
        tok.save_pretrained(out_dir)
        import shutil
        shutil.rmtree(ckpt, ignore_errors=True)
        _, pv = run_eval(model, dl_va)
        _, pt = run_eval(model, dl_te)
        print(f"    seed {init_seed} DONE best_ep={best_ep} "
              f"test_ROC-AUC={roc_auc_score(y_te, pt):.4f} ({time.time()-t0:.0f}s)", flush=True)
        return pv, pt, best_ep

    os.makedirs(ENS_DIR, exist_ok=True)
    os.makedirs(PROBS_DIR, exist_ok=True)
    val_probs, test_probs, per_seed = [], [], []
    t_start = time.time()
    for s in args.seeds:
        print(f"\n=== Training seed {s} ({args.seeds.index(s)+1}/{len(args.seeds)}) ===", flush=True)
        out_dir = os.path.join(ENS_DIR, f"seed_{s}")
        pv, pt, best_ep = train_one(s, out_dir)
        np.save(os.path.join(PROBS_DIR, f"seed_{s}_val.npy"), pv)
        np.save(os.path.join(PROBS_DIR, f"seed_{s}_test.npy"), pt)
        val_probs.append(pv)
        test_probs.append(pt)
        per_seed.append({"seed": s, "best_epoch": best_ep,
                         "test_roc_auc": float(roc_auc_score(y_te, pt)),
                         "test_pr_auc": float(average_precision_score(y_te, pt))})

    # ---- soft-voting ensemble ----
    pv_ens = np.mean(val_probs, axis=0)
    pt_ens = np.mean(test_probs, axis=0)
    t_f1, _ = best_threshold(y_va, pv_ens, beta=1.0)
    t_f2, _ = best_threshold(y_va, pv_ens, beta=2.0)
    wss, t_r95, _ = wss_at_recall(y_va, pv_ens, args.target_recall)

    def test_metrics(thr):
        pred = (pt_ens >= thr).astype(int)
        return {
            "threshold": round(float(thr), 4),
            "accuracy": round(float(accuracy_score(y_te, pred)), 4),
            "precision": round(float(precision_score(y_te, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_te, pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_te, pred, zero_division=0)), 4),
        }

    test_wss, _, _ = wss_at_recall(y_te, pt_ens, args.target_recall)
    meta = {
        "ensemble": True,
        "n_models": len(args.seeds),
        "seeds": args.seeds,
        "base_model": args.model,
        "max_len": args.max_len,
        "freeze_layers": args.freeze_layers,
        "split_seed": args.split_seed,
        "threshold_f1": round(float(t_f1), 4),
        "threshold_f2": round(float(t_f2), 4),
        "threshold_recall95": round(float(t_r95), 4),
        "deploy_threshold": "threshold_f2",
        "target_recall": args.target_recall,
        "roc_auc": round(float(roc_auc_score(y_te, pt_ens)), 4),
        "pr_auc": round(float(average_precision_score(y_te, pt_ens)), 4),
        "wss_test": round(float(test_wss), 4),
        "metrics_test_f1": test_metrics(t_f1),
        "metrics_test_f2": test_metrics(t_f2),
        "per_seed": per_seed,
        "train_seconds": round(time.time() - t_start, 1),
    }
    with open(ENSEMBLE_META_PATH := os.path.join(ENS_DIR, "ensemble_meta.json"),
              "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\n================ ENSEMBLE (held-out TEST) ================")
    print(f"  ROC-AUC : {meta['roc_auc']*100:.1f}%")
    print(f"  PR-AUC  : {meta['pr_auc']*100:.1f}%")
    print(f"  WSS@95  : {meta['wss_test']*100:.1f}%")
    print(f"  F2-opt (deploy) thr={t_f2:.3f} -> "
          f"recall {meta['metrics_test_f2']['recall']*100:.1f}% "
          f"precision {meta['metrics_test_f2']['precision']*100:.1f}%")
    print(f"  F1-opt          thr={t_f1:.3f} -> "
          f"recall {meta['metrics_test_f1']['recall']*100:.1f}% "
          f"precision {meta['metrics_test_f1']['precision']*100:.1f}%")
    print(f"  saved -> {ENSEMBLE_META_PATH}")
    print(f"  total train time: {meta['train_seconds']/60:.1f} min")
    print("==========================================================")


if __name__ == "__main__":
    main()
