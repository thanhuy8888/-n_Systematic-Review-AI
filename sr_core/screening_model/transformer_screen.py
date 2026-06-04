"""
Semantic screening engine - hybrid PubMedBERT + XGBoost (Chapter 3.4 / 4.3
of the thesis).

Pipeline:
    text -> PubMedBERT -> 768-d embedding
                       +
              TF-IDF (max 2000 feats, 1-2 grams)
                       v
                    hstack -> SMOTE (at train time)
                       v
              XGBoost + Random Forest VotingClassifier (soft, w=[2,1])
                       v
              include / exclude probability

At inference time this module loads the persisted hybrid classifier from
hybrid_xgb_model.pkl and the TF-IDF vectorizer from tfidf_vectorizer.pkl
(both produced by experiments/baselines/evaluate_transformer.py).

If either artifact is missing (fresh checkout with no training run yet), the
screener falls back to a PubMedBERT-only PICO-similarity scorer so the API
stays functional, and the rationale string records the degraded mode.

Public API (unchanged for backward compatibility with apps/api/routers/screening.py):
    TransformerScreeningModel
    get_screening_model()
    predict(title, abstract, criteria, threshold) -> dict
"""
import os
import re
from typing import Dict, List, Optional

import numpy as np

# Optional heavy dependencies - imported lazily so the package still imports
# in CI / environments without GPU drivers.
try:
    import torch  # noqa: F401
    _TORCH_OK = True
except Exception:
    _TORCH_OK = False

try:
    from sentence_transformers import SentenceTransformer
    _ST_OK = True
except Exception:
    SentenceTransformer = None  # type: ignore
    _ST_OK = False

try:
    import joblib
    _JOBLIB_OK = True
except Exception:
    _JOBLIB_OK = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# PubMedBERT sentence-transformer checkpoint used by
# experiments/baselines/evaluate_transformer.py. Keeping the same checkpoint
# is what makes the persisted XGBoost classifier reusable at inference time.
MODEL_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"

_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
HYBRID_CLF_PATH = os.path.join(_MODEL_DIR, "hybrid_xgb_model.pkl")
TFIDF_PATH = os.path.join(_MODEL_DIR, "tfidf_vectorizer.pkl")


class TransformerScreeningModel:
    """
    Hybrid PubMedBERT + XGBoost screener.

    The constructor tries to load every component; missing components only
    degrade the mode (full hybrid -> embedding-similarity fallback) without
    raising, so the API can still serve traffic on a fresh checkout.
    """

    def __init__(self, model_name: str = MODEL_NAME, device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if _TORCH_OK and torch.cuda.is_available() else "cpu")
        self.embedder: Optional[SentenceTransformer] = None
        self.classifier = None
        self.vectorizer = None
        self.mode = "fallback"  # set to "hybrid" only when every artifact loads

        self._load_embedder()
        self._load_hybrid_artifacts()

        print(
            f"[Screening] PubMedBERT loaded={self.embedder is not None}, "
            f"classifier loaded={self.classifier is not None}, "
            f"vectorizer loaded={self.vectorizer is not None}, mode={self.mode}"
        )

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    def _load_embedder(self) -> None:
        if not _ST_OK:
            print("[Screening] sentence-transformers not installed - embeddings disabled.")
            return
        try:
            print(f"[Screening] Loading PubMedBERT ({self.model_name}) on {self.device}...")
            self.embedder = SentenceTransformer(self.model_name, device=self.device)
        except Exception as e:
            print(f"[Screening] Warning: could not load {self.model_name}: {e}")
            self.embedder = None

    def _load_hybrid_artifacts(self) -> None:
        if not _JOBLIB_OK:
            return
        try:
            if os.path.exists(HYBRID_CLF_PATH):
                self.classifier = joblib.load(HYBRID_CLF_PATH)
            if os.path.exists(TFIDF_PATH):
                self.vectorizer = joblib.load(TFIDF_PATH)
        except Exception as e:
            print(f"[Screening] Warning: failed to load hybrid artifacts: {e}")
            self.classifier = None
            self.vectorizer = None

        if self.embedder is not None and self.classifier is not None and self.vectorizer is not None:
            self.mode = "hybrid"

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _format_pico_criteria(criteria: Dict[str, str]) -> str:
        parts = []
        if criteria.get("population"):
            parts.append(f"Population: {criteria['population']}")
        if criteria.get("intervention"):
            parts.append(f"Intervention: {criteria['intervention']}")
        if criteria.get("comparison"):
            parts.append(f"Comparison: {criteria['comparison']}")
        if criteria.get("outcome"):
            parts.append(f"Outcome: {criteria['outcome']}")
        if criteria.get("studyType"):
            parts.append(f"Study Type: {criteria['studyType']}")
        return " ".join(parts) if parts else "Include all studies."

    @staticmethod
    def _extract_keywords(criteria: Dict[str, str]) -> List[str]:
        kws: List[str] = []
        for value in criteria.values():
            if not value:
                continue
            for kw in re.split(r"[;,]", value.lower()):
                kw = kw.strip()
                if len(kw) > 1:
                    kws.append(kw)
        return kws

    @staticmethod
    def _best_sentence(abstract: str, matched: List[str]) -> str:
        if not abstract or len(abstract) < 20 or not matched:
            return ""
        best, best_score = "", -1
        for raw in abstract.split(". "):
            sent = raw.strip()
            if len(sent.split()) <= 4:
                continue
            sent_lower = sent.lower()
            score = sum(1 for kw in matched if kw in sent_lower)
            if score > best_score:
                best_score, best = score, sent + "."
        return best

    @staticmethod
    def _decide(prob_include: float, threshold: float):
        is_uncertain = abs(prob_include - threshold) < 0.10
        if is_uncertain:
            return "UNCERTAIN", True
        return ("INCLUDE" if prob_include >= threshold else "EXCLUDE"), False

    # ------------------------------------------------------------------ #
    # Inference - dispatcher
    # ------------------------------------------------------------------ #
    def predict(
        self,
        title: str,
        abstract: str,
        criteria: Dict[str, str],
        threshold: float = 0.5,
    ) -> Dict:
        pico_text = self._format_pico_criteria(criteria)
        paper_text = f"{title}. {abstract}".strip()

        if not pico_text.strip() or pico_text == "Include all studies.":
            return {
                "decision": "INCLUDE",
                "confidence": 1.0,
                "is_uncertain": False,
                "reason": "No strict PICO criteria provided. Auto-inclusion.",
            }

        if self.mode == "hybrid":
            return self._predict_hybrid(paper_text, abstract, criteria, threshold)
        if self.embedder is not None:
            return self._predict_embedding_only(paper_text, abstract, criteria, threshold)
        return self._predict_keyword_only(paper_text, abstract, criteria, threshold)

    # ------------------------------------------------------------------ #
    # Mode 1 - full hybrid (matches thesis Ch. 3.4 / 4.3 numbers)
    # ------------------------------------------------------------------ #
    def _predict_hybrid(
        self,
        paper_text: str,
        abstract: str,
        criteria: Dict[str, str],
        threshold: float,
    ) -> Dict:
        try:
            emb = self.embedder.encode([paper_text], show_progress_bar=False, batch_size=1)
            tfidf = self.vectorizer.transform([paper_text]).toarray()
            X = np.hstack((emb, tfidf))

            if hasattr(self.classifier, "predict_proba"):
                prob_include = float(self.classifier.predict_proba(X)[0, 1])
            else:
                prob_include = float(self.classifier.predict(X)[0])
                prob_include = max(0.0, min(1.0, prob_include))
        except Exception as e:
            print(f"[Screening] Hybrid inference failed ({e}); falling back to embedding mode.")
            return self._predict_embedding_only(paper_text, abstract, criteria, threshold)

        decision, is_uncertain = self._decide(prob_include, threshold)
        matched = [kw for kw in self._extract_keywords(criteria) if kw in paper_text.lower()]
        best_sent = self._best_sentence(abstract, matched)

        rationale = (
            f"Hybrid PubMedBERT + XGBoost score: {prob_include:.0%}.\n"
            f"Matched PICO keywords ({len(matched)}): {', '.join(matched[:8]) or '-'}."
        )
        if best_sent:
            rationale += f"\nEvidence: \"{best_sent}\""
        if is_uncertain:
            rationale += "\nLow margin - flagged for human review."

        return {
            "decision": decision,
            "confidence": prob_include,
            "is_uncertain": is_uncertain,
            "reason": rationale,
        }

    # ------------------------------------------------------------------ #
    # Mode 2 - PubMedBERT-only PICO similarity (classifier missing)
    # ------------------------------------------------------------------ #
    def _predict_embedding_only(
        self,
        paper_text: str,
        abstract: str,
        criteria: Dict[str, str],
        threshold: float,
    ) -> Dict:
        keywords = self._extract_keywords(criteria)
        if not keywords:
            return {
                "decision": "INCLUDE",
                "confidence": 1.0,
                "is_uncertain": False,
                "reason": "No keywords extracted from PICO criteria.",
            }

        text_lower = paper_text.lower()
        matched: List[str] = [kw for kw in keywords if kw in text_lower]

        try:
            paper_emb = self.embedder.encode([paper_text[:2000]], show_progress_bar=False)[0]
        except Exception:
            paper_emb = None

        scores: Dict[str, float] = {}
        for kw in keywords:
            if kw in text_lower:
                scores[kw] = 1.0
                continue
            if paper_emb is None:
                scores[kw] = 0.0
                continue
            try:
                kw_emb = self.embedder.encode([kw], show_progress_bar=False)[0]
                denom = (np.linalg.norm(kw_emb) * np.linalg.norm(paper_emb)) or 1.0
                sim = float(np.dot(kw_emb, paper_emb) / denom)
                scores[kw] = max(0.0, min(1.0, (sim - 0.75) / 0.2))
            except Exception:
                scores[kw] = 0.0

        prob_include = sum(scores.values()) / len(scores) if scores else 0.0
        prob_include = max(0.0, min(1.0, prob_include))

        decision, is_uncertain = self._decide(prob_include, threshold)
        best_sent = self._best_sentence(abstract, matched)

        kw_details = [
            f"{'OK' if s > 0.7 else ('?' if s > 0.3 else 'X')} '{kw}': {s:.0%}"
            for kw, s in scores.items()
        ]
        rationale = (
            f"PubMedBERT similarity score: {prob_include:.0%} "
            f"(fallback mode - hybrid XGBoost artifact not loaded).\n"
            f"Keyword match: {', '.join(kw_details)}."
        )
        if best_sent:
            rationale += f"\nEvidence: \"{best_sent}\""
        if is_uncertain:
            rationale += "\nLow margin - flagged for human review."

        return {
            "decision": decision,
            "confidence": prob_include,
            "is_uncertain": is_uncertain,
            "reason": rationale,
        }

    # ------------------------------------------------------------------ #
    # Mode 3 - pure keyword heuristic (no transformer available)
    # ------------------------------------------------------------------ #
    def _predict_keyword_only(
        self,
        paper_text: str,
        abstract: str,
        criteria: Dict[str, str],
        threshold: float,
    ) -> Dict:
        text = paper_text.lower()
        score = 0.5
        for kw in (criteria.get("intervention") or "").lower().split(","):
            if kw.strip() and kw.strip() in text:
                score += 0.2
        if "mice" in text or "animal" in text:
            score -= 0.3
        score = max(0.0, min(1.0, score))
        decision, is_uncertain = self._decide(score, threshold)
        return {
            "decision": decision,
            "confidence": score,
            "is_uncertain": is_uncertain,
            "reason": (
                "Keyword-heuristic fallback (no PubMedBERT or hybrid classifier "
                f"available). Score = {score:.2f}."
            ),
        }


# ---------------------------------------------------------------------------
# Singleton accessor (used by apps/api/routers/screening.py)
# ---------------------------------------------------------------------------
_screening_pipeline: Optional[TransformerScreeningModel] = None


def get_screening_model() -> TransformerScreeningModel:
    global _screening_pipeline
    if _screening_pipeline is None:
        _screening_pipeline = TransformerScreeningModel()
    return _screening_pipeline
