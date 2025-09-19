import math

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("statsmodels")

from nli_groundedness.mixed_effects import RegressionSpec, fit_model
from nli_groundedness.propensity import average_treatment_effect, fit_propensity, nearest_neighbor_match
from nli_groundedness.regressor_validation import evaluate_predictions
from scripts.build_lowG_mnli import build_subset, compute_balance_checks


def _toy_dataframe(n: int = 32):
    rng = np.random.default_rng(0)
    grounded = rng.uniform(-1, 1, size=n)
    noise = rng.normal(scale=0.1, size=n)
    difficulty = 0.5 * grounded + noise
    data = {
        "groundedness": grounded,
        "difficulty": difficulty,
        "premise_len": rng.integers(8, 16, size=n),
        "hyp_len": rng.integers(3, 8, size=n),
        "lex_overlap": rng.random(size=n),
        "negation": rng.integers(0, 2, size=n),
        "quantifiers": rng.integers(0, 2, size=n),
        "concreteness_mean": rng.random(size=n),
        "wf_mean": rng.random(size=n),
        "noun_share": rng.random(size=n),
        "verb_share": rng.random(size=n),
        "adj_share": rng.random(size=n),
        "readability": rng.random(size=n),
        "genre": ["fiction" if i % 2 else "slate" for i in range(n)],
        "label": ["entailment" if i % 3 else "neutral" for i in range(n)],
        "premise_id": [f"p{i//2}" for i in range(n)],
    }
    return pd.DataFrame(data)


def test_linear_mixed_effects_estimates_coefficient():
    df = _toy_dataframe()
    spec = RegressionSpec(outcome="difficulty", family="gaussian", controls=("premise_len", "hyp_len"), random_effects=())
    result, _ = fit_model(df, spec)
    assert math.isfinite(result.coef_groundedness)
    assert abs(result.coef_groundedness - 0.5) < 0.2


def test_regressor_metrics_basic():
    preds = [0.1, 0.2, 0.3, 0.4]
    gold = [0.2, 0.2, 0.4, 0.5]
    controls = {"len": [1, 2, 3, 4]}
    metrics = evaluate_predictions(preds, gold, controls)
    assert math.isfinite(metrics.mae)
    assert metrics.ece is not None


def test_propensity_matching_pipeline():
    df = pd.DataFrame(
        {
            "low_groundedness": [1, 1, 0, 0],
            "premise_len": [10, 12, 11, 9],
            "model_error": [1.0, 0.0, 0.0, 0.0],
            "genre": ["fiction"] * 4,
            "label": ["neutral"] * 4,
        }
    )
    frame, _ = fit_propensity(df, "low_groundedness", ["premise_len"])
    match = nearest_neighbor_match(frame, "low_groundedness")
    assert match.matched_pairs
    ate = average_treatment_effect(frame, match, "model_error", "low_groundedness", n_boot=10)
    assert "ate" in ate


def test_build_low_grounded_subset():
    df = pd.DataFrame(
        {
            "pair_id": ["a", "b", "c", "d"],
            "premise": ["p"] * 4,
            "hypothesis": ["h"] * 4,
            "pred_groundedness": [0.1, -0.5, 0.2, -0.7],
            "gold_label": ["entailment", "entailment", "neutral", "neutral"],
            "genre": ["slate", "slate", "slate", "government"],
            "premise_len": [12, 13, 15, 11],
        }
    )
    subset, stats = build_subset(df, "pred_groundedness", "gold_label", "genre", pct=0.5)
    assert len(subset) == 2
    checks = compute_balance_checks(df, subset, "gold_label", "genre")
    assert "length_kl" in checks
