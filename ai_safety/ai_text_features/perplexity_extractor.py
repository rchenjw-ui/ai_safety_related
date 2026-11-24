from llama_cpp import Llama
import numpy as np
import time







def get_logprobs(llm, text: bytes | str, add_bos=True):
    if isinstance(text, str):
        text = text.encode("utf-8")
    
    tokens = llm.tokenize(text, add_bos=add_bos)
    if len(tokens) < 2:
        return tokens, np.array([])

    llm.reset()
    llm.eval(tokens)
    print(len(llm.scores))
    logprobs = []
    for i in range(len(tokens) - 1):
        logits = llm.scores[i]
        max_logit = np.max(logits)
        log_prob = logits[tokens[i+1]] - (max_logit + np.log(np.sum(np.exp(logits - max_logit))))
        logprobs.append(log_prob)

    return tokens, np.array(logprobs, dtype=np.float32)

#will need to compute the 

if __name__ == "__main__":
    text = r"""
Threading vs Multiprocessing
Threading: Generally not useful here. The GIL + the fact that the heavy work is in C++/CUDA means threads won’t give you real parallelism, and sharing one Llama instance across threads is risky.
Multiprocessing: The correct tool. Each process gets its own model copy and its own KV cache.
Small extra tipIf you only need the final perplexity number (not per-token logprobs for later analysis), you can make the worker even leaner. But keeping the logprobs (or at least mean logprob + token count) is usually worth it if you later want burstiness or confidence estimates.Want me to also show a version that returns per-sentence perplexities (for the burstiness signal) while still being multiprocessing-friendly?
    """


    llm = Llama(model_path="LFM2.5-230M-IQ4_NL.gguf", 
                n_gpu_layers=-1, 
                n_batch =1024,
                n_ubatch=1024,
                logits_all=True, 
                verbose=False)
    print(len(llm.tokenize(text.encode())))
    print("LOGPROBS")

    start = time.perf_counter()
    tokens, logprobs = get_logprobs(llm, text)
    end = time.perf_counter()
    print("TIME")
    print(end-start)
    ppl = np.exp(-np.mean(logprobs))
    print("Perplexity:", ppl)
    print("Per-token logprobs:", logprobs)

#