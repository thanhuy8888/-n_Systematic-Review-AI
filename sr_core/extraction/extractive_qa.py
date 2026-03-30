import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

MODEL_NAME = "deepset/roberta-base-squad2"

class ExtractiveQAPipeline:
    """
    Structured extraction from full-text or abstracts using Extractive Question-Answering.
    Uses direct AutoModel loading instead of pipeline() for maximum compatibility.
    """
    def __init__(self, model_name: str = MODEL_NAME, device: str = None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[ExtractiveQA] Initializing QA model {model_name} on {self.device}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            self.qa_pipeline = True  # Flag that model is loaded
        except Exception as e:
            print(f"Warning: Could not load {model_name}. {e}")
            self.tokenizer = None
            self.model = None
            self.qa_pipeline = None

    def _answer_question(self, question: str, context: str) -> str:
        """Run QA inference manually using AutoModelForQuestionAnswering."""
        if not self.model or not self.tokenizer:
            return "N/A (Model Not Loaded)"
        
        try:
            # Truncate context to fit model max length
            inputs = self.tokenizer(
                question, context,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                start_logits = outputs.start_logits[0]
                end_logits = outputs.end_logits[0]
            
            input_ids = inputs["input_ids"][0]
            total_tokens = input_ids.shape[0]
            
            # Find where the context starts (after separator tokens)
            sep_positions = (input_ids == self.tokenizer.sep_token_id).nonzero(as_tuple=True)[0]
            if len(sep_positions) >= 2:
                context_start = sep_positions[1].item() + 1
            elif len(sep_positions) == 1:
                context_start = sep_positions[0].item() + 1
            else:
                context_start = 1
            
            # Find end of actual tokens (before padding)
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 1
            non_pad = (input_ids != pad_id).nonzero(as_tuple=True)[0]
            context_end = non_pad[-1].item() if len(non_pad) > 0 else total_tokens - 1
            
            print(f"  [QA] Q: '{question[:50]}' | tokens: {total_tokens} | context range: [{context_start}:{context_end}]")
            
            # Null answer score (CLS position)
            null_score = start_logits[0].item() + end_logits[0].item()
            
            # Mask out all non-context tokens
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
            
            # Limit answer span length (max 80 tokens for academic text)
            if end_idx - start_idx > 80:
                end_idx = start_idx + 80
            
            best_span_score = start_logits[start_idx].item() + end_logits[end_idx].item()
            
            print(f"  [QA] null_score={null_score:.2f} | span_score={best_span_score:.2f} | span=[{start_idx}:{end_idx}]")
            
            # Use a margin: only return "Not Mentioned" if null score is SIGNIFICANTLY higher
            # SQuAD2 models are overly conservative with noisy academic text
            # We increase the null score margin significantly to force extraction of the best matching span
            NULL_SCORE_MARGIN = 15.0
            
            if null_score > best_span_score + NULL_SCORE_MARGIN:
                return "Not Explicitly Mentioned - Requires Manual Review"
            
            answer_ids = input_ids[start_idx:end_idx + 1]
            answer = self.tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
            
            print(f"  [QA] ANSWER: '{answer[:100]}'")
            
            if not answer or len(answer) < 3:
                return "Not Explicitly Mentioned"
            
            # Capitalize first letter if it's a lowercase token
            if answer and answer[0].islower():
                answer = answer[0].upper() + answer[1:]
                
            return answer
            
        except Exception as e:
            print(f"QA Extraction error: {e}")
            import traceback
            traceback.print_exc()
            return "Error extracting"

    def extract_structured_data(self, title: str, text_chunk: str) -> dict:
        """
        Executes multiple QA tasks per paper chunk to populate the Evidence Table.
        """
        full_context = f"{title}. {text_chunk}"
        
        # Limit context length to avoid memory issues, but keep enough for extraction
        if len(full_context) > 3500:
            full_context = full_context[:3500]
        
        # Using more specific keyword-focused queries that work better for scientific extraction
        # rather than conversational questions.
        queries = {
            "methodology": "What was the study design, methodology, or experimental approach?",
            "sampleSize": "How many patients, mice, or participants were included? What was the sample size N?",
            "keyFindings": "What were the main conclusions, results, or key findings of the study?",
            "limitations": "What limitations, weaknesses, or future work were discussed?",
            "riskOfBias": "Was this a randomized, double-blind, cohort, or observational study? What is the risk of bias?"
        }
        
        return {
            "methodology": self._answer_question(queries["methodology"], full_context),
            "sampleSize": self._answer_question(queries["sampleSize"], full_context),
            "keyFindings": self._answer_question(queries["keyFindings"], full_context),
            "limitations": self._answer_question(queries["limitations"], full_context),
            "riskOfBias": self._answer_question(queries["riskOfBias"], full_context),
        }

# Singleton instance
_qa_pipeline = None

def get_qa_pipeline() -> ExtractiveQAPipeline:
    global _qa_pipeline
    if _qa_pipeline is None:
        _qa_pipeline = ExtractiveQAPipeline()
    return _qa_pipeline
