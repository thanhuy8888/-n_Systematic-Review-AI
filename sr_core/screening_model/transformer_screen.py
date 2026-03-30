import os
import torch
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Base model specifically pre-trained on medical literature
MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

class TransformerScreeningModel:
    """
    A Criteria-Aware Transformer model for Systematic Review Screening.
    Instead of just scoring an abstract, it evaluates the logical relationship
    between the PICO criteria and the abstract using Cross-Attention.
    """
    def __init__(self, model_name: str = MODEL_NAME, device: str = None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[TransformerScreening] Initializing model {model_name} on {self.device}...")
        
        # Load tokenizer and model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            # Binary classification: 0 (Exclude), 1 (Include)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Warning: Could not load {model_name} from HuggingFace. {e}")
            self.tokenizer = None
            self.model = None

    def _format_pico_criteria(self, criteria: Dict[str, str]) -> str:
        """
        Flattens the PICO criteria into a single descriptive string for the transformer.
        Target format: "Population: X. Intervention: Y. Comparator: Z. Outcome: W."
        """
        parts = []
        if criteria.get("population"): parts.append(f"Population: {criteria['population']}")
        if criteria.get("intervention"): parts.append(f"Intervention: {criteria['intervention']}")
        if criteria.get("comparison"): parts.append(f"Comparison: {criteria['comparison']}")
        if criteria.get("outcome"): parts.append(f"Outcome: {criteria['outcome']}")
        if criteria.get("studyType"): parts.append(f"Study Type: {criteria['studyType']}")
        
        return " ".join(parts) if parts else "Include all studies."

    def predict(self, title: str, abstract: str, criteria: Dict[str, str], threshold: float = 0.5) -> Dict:
        """
        Evaluates a paper against the criteria using keyword-centric BERT semantic matching.
        """
        # 1. Fallback if model isn't loaded
        if not self.model or not self.tokenizer:
            return self._mock_fallback_prediction(title, abstract, criteria)

        pico_text = self._format_pico_criteria(criteria)
        paper_text = f"{title}. {abstract}"

        if not pico_text.strip() or pico_text == "Include all studies.":
             return {
                 "decision": "INCLUDE",
                 "confidence": 1.0,
                 "is_uncertain": False,
                 "reason": "No strict PICO criteria provided. Auto-inclusion."
             }

        # 2. Keyword-centric scoring: Extract all individual PICO keywords
        all_keywords = []
        for key, value in criteria.items():
            if value.strip():
                for kw in value.lower().replace(';', ',').split(','):
                    kw = kw.strip()
                    if kw and len(kw) > 1:
                        all_keywords.append(kw)
        
        if not all_keywords:
            return {
                "decision": "INCLUDE",
                "confidence": 1.0,
                "is_uncertain": False,
                "reason": "No keywords extracted from PICO criteria."
            }
        
        text_lower = paper_text.lower()
        
        # 3. Score each keyword: exact match + semantic similarity
        keyword_scores = {}
        matched_keywords = []
        base_model = getattr(self.model, getattr(self.model.config, "model_type", "bert"), self.model)
        
        for kw in all_keywords:
            # Exact keyword match — strong evidence
            if kw in text_lower:
                keyword_scores[kw] = 1.0
                matched_keywords.append(kw)
            else:
                # Semantic check: is the concept discussed even if exact word isn't there?
                try:
                    inputs_kw = self.tokenizer(kw, return_tensors="pt", truncation=True, padding=True, max_length=32).to(self.device)
                    inputs_paper = self.tokenizer(paper_text[:1000], return_tensors="pt", truncation=True, padding=True, max_length=512).to(self.device)
                    
                    with torch.no_grad():
                        emb_kw = base_model(**inputs_kw).last_hidden_state[:, 0, :]
                        emb_paper = base_model(**inputs_paper).last_hidden_state[:, 0, :]
                        sim = torch.nn.functional.cosine_similarity(emb_kw, emb_paper).item()
                    
                    # Rescale: cosine in biomedical BERT space is typically 0.7-0.95
                    # Shift so 0.85 maps to ~0.5 (neutral), 0.95 maps to ~1.0 (relevant)
                    rescaled = max(0.0, min(1.0, (sim - 0.75) / 0.2))
                    keyword_scores[kw] = rescaled
                except:
                    keyword_scores[kw] = 0.0
        
        # 4. Final score = weighted average of keyword scores
        if keyword_scores:
            include_prob = sum(keyword_scores.values()) / len(keyword_scores)
        else:
            include_prob = 0.0
        
        include_prob = max(0.0, min(1.0, include_prob))
        
        decision = "INCLUDE" if include_prob >= threshold else "EXCLUDE"
        is_uncertain = abs(include_prob - threshold) < 0.10
        if is_uncertain:
            decision = "UNCERTAIN"

        # 5. XAI: Find the most relevant sentence
        best_sentence = ""
        if abstract and len(abstract) > 20:
            sentences = [s.strip() + "." for s in abstract.split(". ") if len(s.split()) > 4]
            if sentences and matched_keywords:
                # Pick the sentence containing the most matched keywords
                best_score = -1
                for sent in sentences:
                    sent_lower = sent.lower()
                    count = sum(1 for kw in matched_keywords if kw in sent_lower)
                    if count > best_score:
                        best_score = count
                        best_sentence = sent
        
        # 6. Build rationale
        rationale = f"Tỉ lệ phù hợp: {include_prob:.0%}."
        
        # Keyword breakdown
        kw_details = []
        for kw, score in keyword_scores.items():
            status = "✅" if score > 0.7 else ("⚠️" if score > 0.3 else "❌")
            kw_details.append(f"{status} '{kw}': {score:.0%}")
        rationale += f"\n🔑 Keyword Match: {', '.join(kw_details)}."
        
        if best_sentence:
            rationale += f"\n📌 Evidence: \"{best_sentence}\""
            
        if is_uncertain:
            rationale += "\n⚠️ AI đang phân vân, cần Human review."

        return {
            "decision": decision,
            "confidence": include_prob,
            "is_uncertain": is_uncertain,
            "reason": rationale.strip()
        }
        
    def _mock_fallback_prediction(self, title: str, abstract: str, criteria: Dict[str, str]) -> Dict:
        """
        A simulated baseline used when HuggingFace weights cannot be immediately downloaded
        (useful for CI/CD or local dev without 1GB+ model).
        """
        text = (title + " " + abstract).lower()
        score = 0.5
        
        target_keywords = criteria.get("intervention", "").lower().split(",")
        for kw in target_keywords:
            if kw.strip() and kw.strip() in text:
                score += 0.2
        
        if "mice" in text or "animal" in text:
            score -= 0.3
            
        decision = "INCLUDE" if score > 0.6 else "EXCLUDE"
        return {
            "decision": decision,
            "confidence": score,
            "is_uncertain": False,
            "reason": f"Fallback Criteria Match (Simulated Transformer Score = {score:.2f})"
        }

# Singleton instance
_screening_pipeline = None

def get_screening_model() -> TransformerScreeningModel:
    global _screening_pipeline
    if _screening_pipeline is None:
        _screening_pipeline = TransformerScreeningModel()
    return _screening_pipeline
