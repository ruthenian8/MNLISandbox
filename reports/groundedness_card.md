# Groundedness-Aware NLI Benchmarking Card

## Motivation

The benchmark quantifies how strongly each SNLI premise depends on perceptual evidence. High groundedness indicates that the
visual scene makes the premise linguistically plausible, whereas low groundedness exposes examples where language models must
reason beyond surface cues. By correlating these scores with human disagreement (ChaosNLI) and model errors, we can prioritise
cases that challenge both annotators and automated systems. The resulting MNLI-LowG slice offers a fairer stress test for
reasoning-focused large language models.

## Definition
- **Sentence groundedness** is computed as the average token-level pointwise mutual information (PMI) between a caption-conditioned language model and a text-only language model.
- PMI for a token *t* is defined as: `log p(t | image, context) - log p(t | context)`. Positive values indicate that the image raises the probability of observing the token.
- We aggregate PMI over content words (Universal Dependencies POS in {NOUN, PROPN, VERB, ADJ, ADV, NUM, PRON}). The default aggregator is the mean over content words; we also report an uncertainty coefficient defined as the mean relative surprisal reduction.

## Data
- **SNLI** premises paired with **Flickr30k** images supply ground truth groundedness targets.
- **ChaosNLI** provides human label entropies for disagreement analysis.
- **MNLI** receives groundedness predictions from the text-only regressor to enable cross-dataset benchmarking.

All datasets must be downloaded manually following their respective licenses (SNLI/MNLI: CC BY-SA 4.0, Flickr30k: custom academic license, ChaosNLI: CC BY-SA 4.0). Paths are configured via `configs/paths.yaml`.

## Modeling
1. **Captioner vs. Text LM**: We support multimodal checkpoints such as PaliGemma, Gemma-3 VLM, and LLaVA-OneVision. In resource-limited environments, a deterministic stub is used for development and unit tests.
2. **Groundedness Aggregation**: Token PMI and surprisal deltas are aggregated with helper utilities in `nli_groundedness/groundedness.py`. Token-level diagnostics are stored for auditing.
3. **Text-Only Regressor**: The production workflow trains a RoBERTa-base regressor over SNLI groundedness labels and averages predictions over 10 seeds. For lightweight execution we ship a linear fallback that regresses on sentence length.
4. **Binning**: MNLI predictions are discretised into quantile bins (configurable) to facilitate stratified evaluation.
5. **Reasoning Models**: Any instruction-tuned LM can be evaluated via `scripts/07_run_reasoning_models.py`. The helper counts prompt/generation tokens and maps free-form outputs to NLI labels.

## Analysis Artifacts
- `artifacts/snli_groundedness.parquet`: token diagnostics and groundedness scores for SNLI premises.
- `artifacts/snli_chaos_join.parquet`: merges ChaosNLI entropy with groundedness.
- `artifacts/snli_st_eval.parquet`: Sentence Transformer baseline predictions.
- `artifacts/regressor/`: regressor predictions and (optionally) checkpoints.
- `artifacts/mnli_pred_groundedness.parquet`: MNLI predictions with groundedness bins.
- `artifacts/reasoning_runs/`: per-model predictions and token accounting.
- `release/mnli_lowg_ids.txt`: recommended MNLI-LowG split (lowest groundedness quantile) for fair reasoning benchmarking.
- `figures/`: correlation plots, calibration curves, per-bin metrics (generated when matplotlib is available).

## Caveats
- Tokenisation drift between captioner and tokenizer should be monitored; mismatched IDs are rejected.
- High PMI outliers may stem from rare tokens or hallucinated visual content. Inspect token diagnostics when possible.
- The stub implementations bundled with this repository are **not** substitutes for real models; they merely enable reproducible unit tests without heavyweight dependencies.
- License compliance for Flickr30k images is critical—redistribution is not permitted.
- When using reasoning LMs, ensure consistent prompting and deterministic generation (temperature 0) for replicability.

## Responsible Use
- Groundedness scores are heuristic; they should not be used in isolation to judge annotation quality or worker reliability.
- Derived slices such as MNLI-LowG should be shared alongside detailed documentation and license information for downstream use.

## Acknowledgements
- SNLI (Bowman et al., 2015)
- Flickr30k Entities (Plummer et al., 2015)
- ChaosNLI (Nie et al., 2020)
- MNLI (Williams et al., 2017)
