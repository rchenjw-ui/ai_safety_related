| Project | Overview | Topics |
|------|------| ---- |
| `llm_sae` | Sparse autoencoder vs neurons for MechInt |AI Safety, Mechanistic Interpretability, LLMs, NLP|
| `fact_checker` | AI Fact Checker given NLI on Corpus |AI Safety, NLP, LLMs, Fact-checking|
| `ai_image_detector` | Test of diffusion noise feature, based on `https://arxiv.org/abs/2312.02625` |AI Safety, Computer Vision|
| `scipaper_eval` | Evaluates LLM scientific abilities given scientific paper results/evidence | AI Safety, NLP, LLMs|
| `prompt_safety` | Prompt injection detector | AI Safety, NLP, LLMs | 
|`ai_text_ModernBert` |Detects LLM-generated text via fine-tuned ModernBERT|AI Safety, NLP, LLMs |
|`ai_text_features`|Uses perplexity and other features for LLM text detection| AI Safety, NLP, LLMs |
|`rbf_vs_nn`|RBF vs NN on MNIST|ML, Computer Vision, Basic Deep Learning |
|`image_optimizer`|Gradient-based optimization through MNIST CNN|ML, Computer Vision, Basic Deep Learning |
|`tox21_revisited`|self-normalizing neural net vs MLP on feature selected RDKIT/MAACS, chemberta and ECFP| Cheminformatics, deep learning |
|`enzyme_function_classifier`|ESM-2 for enzyme function classifier| Cheminformatics, Bioinformatics, Deep Learning|
|`reaction_predictor`|Neural network for reaction prediction| Cheminformatics, deep learning |
|`chem_vae`|VAE for SELFIES string; see if optimization can be done on latent | Cheminformatics, deep learning|
|`ecfp_reverser`|test ECFP reversal | Cheminformatics, deep learning |

## AI TEXT DETECTION FEATURES

* **Perplexity & Entropy Metrics:**
* Global Mean Perplexity
* Perplexity Variance / Burstiness 
* Min-k Perplexity Percentage (identifying unnatural token predictability)
* Token Surprisal Entropy


* **Stylometric & Structural Features:**
* Sentence Length Standard Deviation 
* Type-Token Ratio (TTR) & Root/Mass TTR adjustments
* Structural formatting density (Markdown headers, punctuation distribution)


* **Rule-Based & Heuristic Signatures (Regex):**
* Tricolons (three-part parallel phrasing lists)
* Epistemic Softening / Hedging (`it is worth noting`, `could be argued`, `in summary`)
* Formulaic Contrast markers (`not merely X, but Y`)
* Flagged AI Cliché N-Grams (`delve into`, `testament to`, `tapestry`, `vital role`)


## Evaluation Metrics

To comprehensively assess model performance across skewed class distributions:

1. **ROC-AUC:** Measures overall ranking quality and separability.
2. **F1 Score:** Balances precision and recall at a standardized threshold ($\tau = 0.5$).
3. **PR-AUC (AU-PRC):** Evaluates performance explicitly under imbalanced datasets where false positives on human text carry high friction.

