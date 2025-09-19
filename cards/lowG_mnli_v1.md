# Low-Groundedness MNLI v1

## Summary
- **What it is:** A slice of MNLI premises predicted to have **low groundedness** by the text-only regressor in `configs/regressor.yaml`. The slice stress-tests abstract or weakly depictable statements where visual priors are unlikely to help.
- **Intended use:** Evaluate robustness of NLI systems on abstract language. We recommend pairing this slice with the full MNLI dev set for calibration-aware benchmarking.
- **Non-guarantees:** The slice does **not** guarantee absence of artifacts or purely reasoning-driven errors. Groundedness is a text-derived proxy and may correlate with other stylistic properties.

## Selection criteria
- Model checkpoint: see `configs/regressor.yaml` (`roberta-base` text encoder with regression head, genre-adversarial option disabled).
- Groundedness scoring: `scripts/score_groundedness.py --data mnli.jsonl --train snli_grounded.jsonl --output release/mnli_groundedness_predictions.jsonl`.
- Thresholding: `scripts/build_lowG_mnli.sh --ckpt roberta_base --pct 25` (per-genre/per-label bottom quartile of predicted groundedness).
- Artifact filter: items where a hypothesis-only baseline confidence ≥ 0.8 are dropped (`--baseline` flag, optional in this sandbox release).

## Statistics
- Size: 2 examples (toy placeholder for the kata environment).
- Genre counts: slate 1, government 1.
- Label counts: entailment 1, neutral 1.
- Length histogram (premise tokens): see `artifacts/lowG_mnli_stats.json` for exact bins.
- Average predicted groundedness per genre:
  - slate: -0.42
  - government: -0.38

## Validity checks
- Calibration of the regressor against a small MNLI pilot: run `scripts/validate_regressor.py --data pilot.jsonl --output artifacts/regressor_validation.json`.
- Correlation with concreteness: use `nli_groundedness.regressor_validation.partial_correlation` controlling for mean concreteness and length (see validation JSON file).
- Length / genre balance: `scripts/build_lowG_mnli.py` reports max absolute deviations (< 5%) and KL divergence (< 0.05 when run on the full dataset).

## Baseline results (fixed decoding, temperature 0.2, top-p 0.95)
| Model | Rationale budget | Accuracy ±95% CI |
|-------|------------------|------------------|
| MNLI classifier (no CoT) | – | placeholder (run scripts/evaluate_reasoning.py) |
| CoT model | 64 tokens | placeholder |
| CoT model | 256 tokens | placeholder |

Use `configs/decoding.json` with `scripts/evaluate_reasoning.py` to populate the table once real runs are available.

## Known limitations & warnings
- Predictions rely on a text-only proxy; items may still reference concrete imagery.
- Placeholder statistics here reflect the lightweight kata environment. Re-run the pipeline on full MNLI before public release.
- Residual hypothesis-only artifacts may persist if the baseline screen is skipped.

## Reproduction
```
PYTHONPATH=. \
  scripts/score_groundedness.py --data data/mnli_dev_matched.jsonl \
  --train data/snli_grounded.jsonl --output release/mnli_groundedness_predictions.jsonl

bash scripts/build_lowG_mnli.sh --ckpt roberta_base --pct 25 --pred release/mnli_groundedness_predictions.jsonl
```
