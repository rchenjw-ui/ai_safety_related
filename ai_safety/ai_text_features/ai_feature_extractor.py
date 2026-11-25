from multiprocessing import Pool, cpu_count
from perplexity_extractor import AIFFeatureExtractor
from datasets import load_dataset
import pandas as pd

# Global variable restricted to each worker process
worker_extractor = None

def init_worker(model_path):
    """Initializes a separate model instance inside each worker process."""
    global worker_extractor
    worker_extractor = AIFFeatureExtractor(model_path=model_path)

def extract_single(text_item):
    """Worker function executed in parallel."""
    global worker_extractor
    return worker_extractor.extract_features(text_item)

if __name__ == "__main__":
    dataset = load_dataset("silentone0725/ai-human-text-detection-v1")
    data = dataset["test"].to_pandas()
    
    # Grab your texts
    texts = list(data["text"])[:50]  # Slice a small test batch first!
    labels = list(data["label"])[:50]
    
    # Set safe worker count (don't overload VRAM if offloading layers)
    num_workers = min(2, cpu_count()) 
    
    print(f"Starting multiprocessing pool with {num_workers} workers...")
    
    with Pool(processes=num_workers, initializer=init_worker, initargs=("LFM2.5-230M-IQ4_NL.gguf",)) as pool:
        results = pool.map(extract_single, texts)
        
