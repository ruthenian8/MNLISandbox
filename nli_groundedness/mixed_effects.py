"""Mixed-effects regression utilities for groundedness studies.

This module provides light-weight wrappers around :mod:`statsmodels` to run
mixed-effects regressions that mirror the specifications described in the
project doc.  The helpers intentionally avoid hard dependencies so they can be
imported in minimal environments – a :class:`RuntimeError` is raised when the
optional packages (``pandas``/``statsmodels``) are not available at runtime.

The functions focus on *templates*: we construct design matrices, evaluate
variants for specification-curve analyses, and expose a small collection of
post-fit diagnostics such as partial :math:`R^2`, variance-inflation factors,
random-effect variances, and residual tables.  Actual datasets can be plugged in
via :class:`pandas.DataFrame` objects or any mapping-like iterable.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import statsmodels.api as sm  # type: ignore
    import statsmodels.formula.api as smf  # type: ignore
    from statsmodels.genmod.cov_struct import Exchangeable  # type: ignore
    from statsmodels.genmod.generalized_estimating_equations import GEE  # type: ignore
    from statsmodels.stats.outliers_influence import variance_inflation_factor  # type: ignore
except Exception:  # pragma: no cover
    sm = None  # type: ignore
    smf = None  # type: ignore
    GEE = None  # type: ignore
    Exchangeable = None  # type: ignore
    variance_inflation_factor = None  # type: ignore


DEFAULT_CONTROLS: Tuple[str, ...] = (
    "premise_len",
    "hyp_len",
    "lex_overlap",
    "negation",
    "quantifiers",
    "concreteness_mean",
    "wf_mean",
    "noun_share",
    "verb_share",
    "adj_share",
    "readability",
)

CONTROL_GROUPS: Dict[str, Tuple[str, ...]] = {
    "length": ("premise_len", "hyp_len"),
    "lexical": ("lex_overlap",),
    "polarity": ("negation", "quantifiers"),
    "norms": ("concreteness_mean", "wf_mean"),
    "pos": ("noun_share", "verb_share", "adj_share"),
    "readability": ("readability",),
}


@dataclass
class RegressionSpec:
    """Configuration for a single mixed-effects model run."""

    outcome: str
    family: str = "gaussian"  # "gaussian" or "binomial"
    controls: Sequence[str] = DEFAULT_CONTROLS
    add_genre: bool = False
    add_label: bool = False
    random_effects: Sequence[str] = ()
    cluster: Optional[str] = None
    weight: Optional[str] = None


@dataclass
class RegressionResult:
    """Summary information for a fitted model."""

    formula: str
    coef_groundedness: float
    std_err: float
    stat: float
    p_value: float
    partial_r2: Optional[float]
    n_obs: int
    variant: str
    controls: Tuple[str, ...]


def _require_dependencies():
    if sm is None or smf is None:
        raise RuntimeError(
            "statsmodels is required for the mixed-effects helpers. "
            "Install statsmodels to run the regression templates."
        )
    if pd is None:
        raise RuntimeError("pandas is required for the mixed-effects helpers.")
    if np is None:
        raise RuntimeError("numpy is required for the mixed-effects helpers.")


def _as_dataframe(data) -> "pd.DataFrame":
    _require_dependencies()
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, Mapping):
        return pd.DataFrame([data])
    return pd.DataFrame(list(data))


def _build_formula(spec: RegressionSpec, include_groundedness: bool = True) -> str:
    pieces: List[str] = []
    if include_groundedness:
        pieces.append("groundedness")
    pieces.extend(spec.controls)
    if spec.add_genre:
        pieces.append("C(genre)")
    if spec.add_label:
        pieces.append("C(label)")
    rhs = " + ".join(pieces) if pieces else "1"
    return f"{spec.outcome} ~ {rhs}"


def _fit_linear_mixed(
    df: "pd.DataFrame",
    formula: str,
    random_effects: Sequence[str],
    weights_col: Optional[str] = None,
):
    groups = random_effects[0] if random_effects else None
    vc_formula = None
    if len(random_effects) > 1:
        vc_formula = {name: f"0 + C({name})" for name in random_effects[1:]}
    if groups is None:
        ols = smf.ols(formula, data=df)
        return ols.fit()
    model = smf.mixedlm(
        formula,
        data=df,
        groups=df[groups],
        vc_formula=vc_formula,
        re_formula="1",
        weights=df[weights_col] if weights_col else None,
    )
    return model.fit(method="lbfgs", disp=False)


def _fit_binomial(
    df: "pd.DataFrame",
    formula: str,
    random_effects: Sequence[str],
    cluster: Optional[str] = None,
    weights_col: Optional[str] = None,
):
    if GEE is None:
        raise RuntimeError(
            "statsmodels >= 0.13 with GEE support is required for binomial mixed models"
        )
    group_col = random_effects[0] if random_effects else cluster
    if group_col is None:
        # fallback to simple GLM with robust variance
        model = smf.glm(formula, data=df, family=sm.families.Binomial())
        result = model.fit(cov_type="cluster", cov_kwds={"groups": df[cluster]}) if cluster else model.fit()
        return result
    cov_struct = Exchangeable()
    model = smf.gee(
        formula,
        groups=df[group_col],
        data=df,
        family=sm.families.Binomial(),
        cov_struct=cov_struct,
    )
    if weights_col:
        result = model.fit(scale=df[weights_col])
    else:
        result = model.fit()
    return result


def _partial_r2(full_result, reduced_result) -> Optional[float]:
    if full_result is None or reduced_result is None:
        return None
    if hasattr(full_result, "ssr") and hasattr(reduced_result, "ssr"):
        ssr_full = float(getattr(full_result, "ssr"))
        ssr_reduced = float(getattr(reduced_result, "ssr"))
    elif hasattr(full_result, "deviance") and hasattr(reduced_result, "deviance"):
        ssr_full = float(getattr(full_result, "deviance"))
        ssr_reduced = float(getattr(reduced_result, "deviance"))
    elif hasattr(full_result, "llf") and hasattr(reduced_result, "llf"):
        # deviance-like metric from log-likelihood
        ssr_full = -2.0 * float(getattr(full_result, "llf"))
        ssr_reduced = -2.0 * float(getattr(reduced_result, "llf"))
    else:
        return None
    if ssr_reduced == 0:
        return None
    return max(0.0, min(1.0, (ssr_reduced - ssr_full) / ssr_reduced))


def fit_model(
    data,
    spec: RegressionSpec,
    variant: str = "full",
    include_groundedness: bool = True,
    controls_override: Optional[Sequence[str]] = None,
):
    """Fit a single mixed-effects model and return the coefficient summary."""

    df = _as_dataframe(data)
    controls = tuple(controls_override or spec.controls)
    working_spec = RegressionSpec(
        outcome=spec.outcome,
        family=spec.family,
        controls=controls,
        add_genre=spec.add_genre,
        add_label=spec.add_label,
        random_effects=spec.random_effects,
        cluster=spec.cluster,
        weight=spec.weight,
    )
    formula = _build_formula(working_spec, include_groundedness=include_groundedness)
    if working_spec.family == "gaussian":
        result = _fit_linear_mixed(df, formula, working_spec.random_effects, working_spec.weight)
    elif working_spec.family == "binomial":
        result = _fit_binomial(df, formula, working_spec.random_effects, working_spec.cluster, working_spec.weight)
    else:
        raise ValueError(f"Unsupported family: {working_spec.family}")

    coef = float(result.params.get("groundedness", 0.0)) if include_groundedness else 0.0
    se = float(result.bse.get("groundedness", float("nan"))) if include_groundedness else float("nan")
    stat_key = "tvalues" if hasattr(result, "tvalues") else "zvalues"
    stat_dict = getattr(result, stat_key, {})
    if isinstance(stat_dict, dict):
        stat = float(stat_dict.get("groundedness", float("nan"))) if include_groundedness else float("nan")
    else:
        stat = float(stat_dict)
    pvalues = getattr(result, "pvalues", {})
    if isinstance(pvalues, Mapping):
        p_value = float(pvalues.get("groundedness", float("nan"))) if include_groundedness else float("nan")
    else:
        p_value = float(pvalues)
    return RegressionResult(
        formula=formula,
        coef_groundedness=coef,
        std_err=se,
        stat=stat,
        p_value=p_value,
        partial_r2=None,
        n_obs=int(getattr(result, "nobs", len(df))),
        variant=variant,
        controls=controls,
    ), result


def specification_curve(
    data,
    spec: RegressionSpec,
    drop_sets: Optional[Sequence[Sequence[str]]] = None,
    include_empty: bool = True,
) -> Tuple[List[RegressionResult], Dict[str, object]]:
    """Run a specification curve for the groundedness coefficient.

    ``drop_sets`` can be used to define bundles of controls that are removed in
    each iteration (e.g. dropping all POS-related features together).  If not
    provided we automatically generate a small library consisting of: the full
    model, models minus each control group in :data:`CONTROL_GROUPS`, and an
    intercept-only baseline.
    """

    df = _as_dataframe(data)
    results: List[RegressionResult] = []
    diagnostics: Dict[str, object] = {}
    base_controls = tuple(spec.controls)

    if drop_sets is None:
        drop_sets = [()] + [CONTROL_GROUPS[name] for name in CONTROL_GROUPS]
        if include_empty:
            drop_sets.append(base_controls)

    for drop in drop_sets:
        remaining = tuple(col for col in base_controls if col not in set(drop))
        variant_name = "full" if not drop else f"minus_{'_'.join(drop)}"
        include_groundedness = True
        reg_result, fitted = fit_model(df, spec, variant=variant_name, include_groundedness=include_groundedness, controls_override=remaining)
        reduced, reduced_fit = fit_model(df, spec, variant=f"{variant_name}_reduced", include_groundedness=False, controls_override=remaining)
        reg_result.partial_r2 = _partial_r2(fitted, reduced_fit)
        results.append(reg_result)
        diagnostics[variant_name] = {
            "residuals": residual_table(df, fitted, spec.outcome),
            "random_effects": random_effect_variance(fitted),
        }
    return results, diagnostics


def vif_table(data, columns: Sequence[str]) -> "pd.DataFrame":
    _require_dependencies()
    df = _as_dataframe(data)
    cols = [col for col in columns if col in df.columns]
    if not cols:
        return pd.DataFrame(columns=["feature", "vif"])
    X = df[cols].assign(_intercept=1.0)
    values = []
    for idx, col in enumerate(cols):
        if variance_inflation_factor is None:
            vif_val = float("nan")
        else:
            vif_val = float(variance_inflation_factor(X.values, idx))
        values.append({"feature": col, "vif": vif_val})
    return pd.DataFrame(values)


def residual_table(df: "pd.DataFrame", fitted_model, outcome: str) -> "pd.DataFrame":
    _require_dependencies()
    resid = getattr(fitted_model, "resid", None)
    if resid is None:
        return pd.DataFrame({})
    frame = df[[outcome]].copy()
    frame["fitted"] = getattr(fitted_model, "fittedvalues", [float("nan")] * len(frame))
    frame["residual"] = list(resid) if isinstance(resid, Iterable) else resid
    return frame


def random_effect_variance(fitted_model) -> Dict[str, float]:
    if hasattr(fitted_model, "cov_re") and fitted_model.cov_re is not None:
        cov = fitted_model.cov_re
        if hasattr(cov, "diag"):
            diag = cov.diagonal()
        elif hasattr(cov, "values"):
            diag = np.array(cov.values).diagonal() if np is not None else []
        else:
            diag = []
        return {f"re_{i}": float(val) for i, val in enumerate(diag)}
    if hasattr(fitted_model, "scale"):
        return {"scale": float(fitted_model.scale)}
    return {}


def specification_grid(spec: RegressionSpec, max_controls: int = 4) -> List[Tuple[str, ...]]:
    """Generate a small grid of control subsets for robustness checks."""

    controls = tuple(spec.controls)
    grid: List[Tuple[str, ...]] = []
    for r in range(1, min(max_controls, len(controls)) + 1):
        for combo in combinations(controls, r):
            grid.append(combo)
    return grid
