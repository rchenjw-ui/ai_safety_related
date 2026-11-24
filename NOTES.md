##  `LLM_SAE` 


##  `FACT_CHECKER` 


##  `AI_IMAGE_DETECTOR` 


##  `SCIPAPER_EVAL` 


##  `PROMPT_SAFETY` 


## `AI_TEXT_MODERNBERT` 


## `AI_TEXT_FEATURES`


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




## `RBF_VS_NN`


## `IMAGE_OPTIMIZER`


## `TOX21_REVISITED`


## `ENZYME_FUNCTION_CLASSIFIER`


## `REACTION_PREDICTOR`


## `CHEM_VAE`


## `ECFP_REVERSER`


(other notes: conways game of life convolution, conv2d, RBM supervised using outcome as hidden of final, forwards backwards)