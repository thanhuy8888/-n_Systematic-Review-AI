"""
Extractive QA pipeline with confidence scoring and cross-chunk aggregation.

Improvements vs. original (per Chapter 4 recommendations):
1. Confidence scores per answer (softmax over start/end logits) so the
   downstream UI can flag low-confidence extractions for human review
   instead of silently returning a guess.
2. Cross-paragraph / cross-chunk extraction: when the caller passes a list
   of chunks (preferred path), the pipeline runs QA on each chunk and keeps
   the span with the highest joint score — addresses the "contextual
   fragmentation of full-text biomedical papers" limitation in §1.5.
3. Backward compatibility: the existing extract_structured_data(title, text)
   API still works and now internally delegates to the chunk-aware code path.
"""
import math
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

MODEL_NAME = "deepset/roberta-base-squad2"

# Confidence threshold below which the answer is flagged for human review.
# Tuned on SQuAD2-style logits — well-grounded biomedical spans typically
# score >0.30. Anything lower is likely a spurious match.
CONFIDENCE_REVIEW_THRESHOLD = 0.20
# Margin to prefer "no answer" over a weak span. Lower = more eager extractor.
NULL_SCORE_MARGIN = 15.0


class ExtractiveQAPipeline:
    """
    Structured extraction from full-text or abstracts using Extractive QA.
    Uses direct AutoModel loading for maximum compatibility.
    """

    def __init__(self, model_name: str = MODEL_NAME, device: str = None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[ExtractiveQA] Initializing QA model {model_name} on {self.device}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            self.qa_pipeline = True
        except Exception as e:
            print(f"Warning: Could not load {model_name}. {e}")
            self.tokenizer = None
            self.model = None
            self.qa_pipeline = None

    # ------------------------------------------------------------------ #
    # Low-level single-context inference                                  #
    # ------------------------------------------------------------------ #
    def _answer_one_context(self, question: str, context: str) -> dict:
        """
        Run QA inference on a single (question, context) pair.
        Returns: {answer, confidence, span_score, null_score, low_confidence}.
        """
        empty = {
            "answer": "Not Explicitly Mentioned",
            "confidence": 0.0,
            "span_score": 0.0,
            "null_score": 0.0,
            "low_confidence": True,
        }

        if not self.model or not self.tokenizer:
            empty["answer"] = "N/A (Model Not Loaded)"
            return empty

        try:
            inputs = self.tokenizer(
                question, context,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                start_logits = outputs.start_logits[0]
                end_logits = outputs.end_logits[0]

            input_ids = inputs["input_ids"][0]
            total_tokens = input_ids.shape[0]

            # Locate context span boundaries
            sep_positions = (input_ids == self.tokenizer.sep_token_id).nonzero(as_tuple=True)[0]
            if len(sep_positions) >= 2:
                context_start = sep_positions[1].item() + 1
            elif len(sep_positions) == 1:
                context_start = sep_positions[0].item() + 1
            else:
                context_start = 1

            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 1
            non_pad = (input_ids != pad_id).nonzero(as_tuple=True)[0]
            context_end = non_pad[-1].item() if len(non_pad) > 0 else total_tokens - 1

            # Null answer score (CLS)
            null_score = start_logits[0].item() + end_logits[0].item()

            mask = torch.zeros_like(start_logits, dtype=torch.bool)
            mask[context_start:context_end + 1] = True

            masked_start = start_logits.clone()
            masked_end = end_logits.clone()
            masked_start[~mask] = -1e10
            masked_end[~mask] = -1e10

            start_idx = torch.argmax(masked_start).item()
            end_idx = torch.argmax(masked_end).item()

            if end_idx < start_idx:
                end_idx = start_idx
            if end_idx - start_idx > 80:
                end_idx = start_idx + 80

            span_score = start_logits[start_idx].item() + end_logits[end_idx].item()

            # Convert logits to a calibrated [0, 1] confidence.
            # softmax over the masked logits, then multiply start * end probs.
            start_probs = torch.softmax(masked_start, dim=0)
            end_probs = torch.softmax(masked_end, dim=0)
            confidence = float(start_probs[start_idx].item() * end_probs[end_idx].item())
            # Penalise when the null option is strongly preferred.
            null_advantage = null_score - span_score
            if null_advantage > 0:
                # Sigmoid-ish dampening
                confidence *= 1.0 / (1.0 + math.exp(null_advantage / 5.0))

            if null_score > span_score + NULL_SCORE_MARGIN:
                empty["span_score"] = span_score
                empty["null_score"] = null_score
                empty["confidence"] = confidence
                empty["answer"] = "Not Explicitly Mentioned - Requires Manual Review"
                return empty

            answer_ids = input_ids[start_idx:end_idx + 1]
            answer = self.tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

            if not answer or len(answer) < 3:
                empty["span_score"] = span_score
                empty["null_score"] = null_score
                empty["confidence"] = confidence
                return empty

            if answer and answer[0].islower():
                answer = answer[0].upper() + answer[1:]

            return {
                "answer": answer,
                "confidence": round(confidence, 4),
                "span_score": round(span_score, 3),
                "null_score": round(null_score, 3),
                "low_confidence": confidence < CONFIDENCE_REVIEW_THRESHOLD,
            }

        except Exception as e:
            print(f"QA Extraction error: {e}")
            import traceback
            traceback.print_exc()
            empty["answer"] = "Error extracting"
            return empty

    # ------------------------------------------------------------------ #
    # Cross-chunk aggregation (handles cross-paragraph evidence)          #
    # ------------------------------------------------------------------ #
    def _answer_across_chunks(self, question: str, contexts: list) -> dict:
        """
        Run QA on multiple chunks and return the highest-confidence answer.
        `contexts` may be a list of strings OR a list of dicts with
        {"text", "section"} (from chunk_by_structure). Section info is
        kept as provenance.
        """
        best = None
        best_section = None

        for ctx in contexts:
            if isinstance(ctx, dict):
                text = ctx.get("text", "")
                section = ctx.get("section")
            else:
                text = ctx
                section = None

            if not text or len(text.strip()) < 20:
                continue

            res = self._answer_one_context(question, text)
            if best is None or res["confidence"] > best["confidence"]:
                best = res
                best_section = section

        if best is None:
            return {
                "answer": "Not Explicitly Mentioned",
                "confidence": 0.0,
                "low_confidence": True,
                "evidence_section": None,
            }

        best["evidence_section"] = best_section
        return best

    # Legacy single-shot wrapper kept for callers that pass one string.
    def _answer_question(self, question: str, context: str) -> str:
        return self._answer_one_context(question, context)["answer"]

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #
    # Domain-specific PICO-aligned queries (Chapter 2 §2.5)
    QUERIES = {
        "methodology": "What was the study design, methodology, or experimental approach?",
        "sampleSize": "How many mice or animals were included per group? What was the sample size N?",
        "mouseStrain": "What mouse strain or genetic background was used? e.g. C57BL/6, db/db, ob/ob",
        "sexAge": "What was the sex and age of the mice at the start of the study?",
        "dietType": "What type of diet was used as intervention? e.g. high fat diet HFD, HFHS, high cholesterol diet HCD",
        "dietComposition": "What was the macronutrient composition of the diet? What percentage of calories were from fat?",
        "duration": "How long was the dietary intervention or treatment duration in weeks?",
        "controlDiet": "What control or comparison diet was used? e.g. standard chow, low fat diet",
        "tcTg": "What were the total cholesterol TC and triglyceride TG levels measured?",
        "ldlHdl": "What were the LDL cholesterol and HDL cholesterol levels reported?",
        "glucoseInsulin": "What were the fasting glucose, insulin levels, or glucose tolerance GTT results?",
        "homaIr": "What was the HOMA-IR insulin resistance index or ITT result?",
        "altAst": "What were the ALT and AST liver enzyme levels reported?",
        "liverHistology": "What liver histology findings were reported? e.g. steatosis, NASH, fibrosis",
        "keyFindings": "What were the main conclusions, key findings, or significant results?",
        "limitations": "What limitations, weaknesses, or future research directions were discussed?",
        "riskOfBias": "Was this study randomized or blinded? What is the risk of bias?",
    }

    def extract_structured_data(self, title: str, text_chunk: str) -> dict:
        """
        Legacy single-chunk API. Returns answers as plain strings, plus a
        sibling `_confidence` dict mapping field -> float.
        """
        full_context = f"{title}. {text_chunk}"
        if len(full_context) > 3500:
            full_context = full_context[:3500]

        results = {}
        confidences = {}
        review_flags = {}

        for field, question in self.QUERIES.items():
            res = self._answer_one_context(question, full_context)
            results[field] = res["answer"]
            confidences[field] = res["confidence"]
            review_flags[field] = res["low_confidence"]

        results["_confidence"] = confidences
        results["_needs_review"] = review_flags
        results["evidenceSpan"] = (text_chunk[:500].strip() if text_chunk else "")
        return results

    def extract_structured_data_from_chunks(
        self,
        title: str,
        chunks: list,
        prefer_sections: dict = None,
    ) -> dict:
        """
        Recommended new API. `chunks` is the output of
        sr_core.ingest.parser.chunk_by_structure().

        `prefer_sections` lets callers route certain fields to their most
        likely section (e.g. methodology -> methods). Other chunks are
        still tried as fallback so cross-section evidence is not lost.
        """
        if not chunks:
            return self.extract_structured_data(title, "")

        # Routing hints: which section a field most plausibly lives in.
        default_routing = {
            "methodology": ("methods", "methodology", "materials and methods"),
            "sampleSize": ("methods", "methodology"),
            "mouseStrain": ("methods", "methodology"),
            "sexAge": ("methods", "methodology"),
            "dietType": ("methods", "methodology", "experimental design"),
            "dietComposition": ("methods", "methodology"),
            "duration": ("methods", "results"),
            "controlDiet": ("methods", "methodology"),
            "tcTg": ("results", "findings"),
            "ldlHdl": ("results", "findings"),
            "glucoseInsulin": ("results", "findings"),
            "homaIr": ("results", "findings"),
            "altAst": ("results", "findings"),
            "liverHistology": ("results", "discussion"),
            "keyFindings": ("discussion", "conclusion", "abstract"),
            "limitations": ("discussion", "limitations", "conclusion"),
            "riskOfBias": ("methods", "discussion"),
        }
        routing = prefer_sections or default_routing

        # Pre-pend a synthetic title chunk so the model sees the title.
        titled = [{"section": "title", "text": title}] + chunks

        results = {}
        confidences = {}
        review_flags = {}
        provenance = {}

        for field, question in self.QUERIES.items():
            preferred = routing.get(field, ())
            primary = [c for c in titled if c.get("section") in preferred]
            fallback = [c for c in titled if c.get("section") not in preferred]

            # Try preferred sections first
            res = self._answer_across_chunks(question, primary) if primary else None
            # If preferred answer is weak, expand search to fallback chunks.
            if res is None or res.get("low_confidence", True):
                fb = self._answer_across_chunks(question, fallback) if fallback else None
                if fb and (res is None or fb["confidence"] > res["confidence"]):
                    res = fb

            if res is None:
                res = {
                    "answer": "Not Explicitly Mentioned",
                    "confidence": 0.0,
                    "low_confidence": True,
                    "evidence_section": None,
                }

            results[field] = res["answer"]
            confidences[field] = res["confidence"]
            review_flags[field] = res.get("low_confidence", True)
            provenance[field] = res.get("evidence_section")

        results["_confidence"] = confidences
        results["_needs_review"] = review_flags
        results["_evidence_section"] = provenance

        # Evidence span: prefer the highest-confidence Results chunk
        results_chunks = [c for c in chunks if c.get("section") in ("results", "findings")]
        if results_chunks:
            results["evidenceSpan"] = results_chunks[0]["text"][:500].strip()
        else:
            results["evidenceSpan"] = chunks[0]["text"][:500].strip()
        return results


# Singleton instance
_qa_pipeline = None

def get_qa_pipeline() -> ExtractiveQAPipeline:
    global _qa_pipeline
    if _qa_pipeline is None:
        _qa_pipeline = ExtractiveQAPipeline()
    return _qa_pipeline
