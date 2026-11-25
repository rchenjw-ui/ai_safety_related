import numpy as np
import re
from llama_cpp import Llama

class AIFFeatureExtractor:
    def __init__(self, model_path: str = "LFM2.5-230M-IQ4_NL.gguf", n_ctx: int = 2048):
        print(f"Loading lightweight model from {model_path}...")
        self.llm = Llama(
            model_path=model_path, 
            n_gpu_layers=-1, 
            n_ctx=n_ctx,          # Fixes the 512 token truncation cap
            n_batch=1024,
            n_ubatch=1024,
            logits_all=True,      # Required to extract token-level scores
            verbose=False
        )

    def get_logprobs(self, text: str, add_bos: bool = True):
        if isinstance(text, str):
            text = text.encode("utf-8")
        
        tokens = self.llm.tokenize(text, add_bos=add_bos)
        if len(tokens) < 2:
            return np.array([], dtype=np.float32)

        self.llm.reset()
        self.llm.eval(tokens)
        
        # Extract logits and targets safely
        logits = np.array(self.llm.scores[:len(tokens) - 1], dtype=np.float32)
        targets = np.array(tokens[1:], dtype=np.int32)

        # Vectorized log-softmax stabilization
        max_logits = np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        log_sum_exp = max_logits.squeeze(-1) + np.log(np.sum(exp_logits, axis=-1))
        
        target_logits = np.take_along_axis(logits, targets[:, None], axis=-1).squeeze(-1)
        return target_logits - log_sum_exp

    def extract_features(self, text: str) -> dict:
        logprobs = self.get_logprobs(text)
        
        if len(logprobs) == 0:
            return {
                "perplexity": 0.0, "perplexity_var": 0.0, "min_k_ppl": 0.0,
                "ttr": 0.0, "sentence_len_std": 0.0, "tricolon_count": 0, "softening_count": 0
            }

        # --- 1. Perplexity & Entropy Metrics ---
        mean_nll = -np.mean(logprobs)
        perplexity = float(np.exp(mean_nll))
        perplexity_var = float(np.var(logprobs))
        
        # Min-k% perplexity (average of the bottom 10% predictable tokens)
        sorted_nlls = np.sort(-logprobs)
        k_count = max(1, int(len(sorted_nlls) * 0.1))
        min_k_ppl = float(np.exp(np.mean(sorted_nlls[:k_count])))

        # --- 2. Stylometric Metrics ---
        words = text.split()
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        sent_lens = [len(s.split()) for s in sentences]

        ttr = len(set(words)) / max(len(words), 1)
        sentence_len_std = float(np.std(sent_lens)) if len(sent_lens) > 1 else 0.0

        # --- 3. Regex Heuristics ---
        tricolons = len(re.findall(r'\b[\w\s]+,\s+[\w\s]+,\s+(?:and|or)\s+[\w\s]+\b', text))
        softening = len(re.findall(r'\b(it is worth noting|could be argued|in summary|testament to|delve)\b', text, re.I))

        return {
            "perplexity": perplexity,
            "perplexity_var": perplexity_var,
            "min_k_ppl": min_k_ppl,
            "ttr": ttr,
            "sentence_len_std": sentence_len_std,
            "tricolon_count": tricolons,
            "softening_count": softening
        }

