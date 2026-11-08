"""SHAP explanations of a fitted model or of one fold of a CV result.

``shap_report`` takes either the dict returned by the notebooks'
``cross_validate_model_cached`` (its last fold by default) or a fitted model,
computes SHAP values and draws three plots: bar, beeswarm and waterfall.
``compute_shap`` and ``plot_shap`` are the two halves, so the slow part can
be cached on its own.

The values are on the scale the model was fitted on. The project's pipelines
fit the model inside a ``TransformedTargetRegressor`` with ``np.log``, so a
SHAP value is a contribution to log(price): additive there, and roughly a
relative change in price.
"""

from __future__ import annotations

import gc
import warnings
from collections.abc import Mapping
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn import config_context

AXIS_LABEL = "Impacto en log(precio)"


# ---------------------------------------------------------------------------
# Resolving the model and its features
# ---------------------------------------------------------------------------
def _is_cv_result(source: Any) -> bool:
    return isinstance(source, Mapping) and {
        "estimator",
        "test_indices",
        "train_indices",
    } <= set(source)


def _is_project_pipeline(source: Any) -> bool:
    steps = getattr(source, "named_steps", {})
    return "imputation" in steps and "outliers" in steps


def _model_and_features(pipeline: Any, rows: pd.DataFrame) -> Any:
    """The bare model of a fitted pipeline and ``rows`` as it sees them.

    Same path as the notebooks' ``fit_report``: imputation, then every step
    of the inner pipeline but the model (encoding, amenities, selection).
    """
    inner = pipeline.named_steps["outliers"].regressor_.regressor_
    with config_context(transform_output="pandas"):
        imputed = pipeline.named_steps["imputation"].transform(rows)
        features = inner[:-1].transform(imputed)
    return inner[-1], features


def _sample(
    X: pd.DataFrame, positions: np.ndarray, n: int, rng: np.random.Generator
) -> pd.DataFrame:
    chosen = rng.choice(positions, size=min(n, len(positions)), replace=False)
    return X.iloc[np.sort(chosen)]


def _resolve(
    source: Any,
    X: pd.DataFrame,
    fold: int,
    n_explain: int,
    n_background: int,
    rng: np.random.Generator,
) -> tuple[Any, pd.DataFrame, pd.DataFrame, int | None]:
    """Return (model, rows to explain, background rows, fold used)."""
    if _is_cv_result(source):
        fold = range(len(source["estimator"]))[fold]  # -1 -> last fold
        pipeline = joblib.load(source["estimator"][fold])
        explain = _sample(X, source["test_indices"][fold], n_explain, rng)
        background = _sample(
            X, source["train_indices"][fold], n_background, rng
        )
        model, explain = _model_and_features(pipeline, explain)
        _, background = _model_and_features(pipeline, background)
        del pipeline  # ~2.6 GB of MICE trees; only the model is needed
        gc.collect()
        return model, explain, background, fold

    positions = np.arange(len(X))
    explain = _sample(X, positions, n_explain, rng)
    background = _sample(X, positions, n_background, rng)
    if _is_project_pipeline(source):
        model, explain = _model_and_features(source, explain)
        _, background = _model_and_features(source, background)
        return model, explain, background, None
    return source, explain, background, None


# ---------------------------------------------------------------------------
# 1. SHAP values
# ---------------------------------------------------------------------------
def compute_shap(
    source: Any,
    X: pd.DataFrame,
    *,
    fold: int = -1,
    n_explain: int = 200,
    n_background: int = 100,
    max_evals: int | None = None,
    random_state: int = 0,
) -> dict:
    """SHAP values of a model, or of one fold of a CV result.

    Parameters
    ----------
    source : one of
        - the dict returned by ``cross_validate_model_cached`` (it needs
          ``estimator``, ``test_indices`` and ``train_indices``). Fold
          ``fold`` is loaded from disk, its test rows are explained and its
          training rows are the background, so the explanations are out of
          sample. The pipeline is freed once the model is taken out.
        - a fitted pipeline of the project's shape (``imputation`` then
          ``outliers``): ``X`` is raw rows, transformed like the model saw
          them.
        - any other fitted regressor: ``X`` is already its feature matrix.
    X : the rows. For a CV result, the same ``X`` that was cross-validated,
        since the fold indices are positions into it.
    fold : which CV fold; ``-1`` is the last one.
    n_explain : rows to explain, sampled at random.
    n_background : rows of the background (the "absent feature" reference).
    max_evals : model evaluations per row for the permutation explainer.
        ``None`` uses its minimum, ``2 * n_features + 1``.
    random_state : seed for the row samples and the permutations.

    Returns
    -------
    dict
        ``explanation`` (``shap.Explanation``), ``features`` (the explained
        rows as the model sees them), ``explainer`` (``"tree"`` or
        ``"permutation"``) and ``fold`` (``None`` unless ``source`` is a CV
        result).

    Notes
    -----
    Tree models shap knows (sklearn forests, XGBoost, LightGBM) get the exact
    ``TreeExplainer``. Anything else, ``BARTAtrRegressor`` included, gets the
    model-agnostic ``PermutationExplainer``, which costs about
    ``n_explain * max_evals * n_background`` predicted rows. Each BART
    prediction runs every posterior draw, so keep those three small.
    """
    rng = np.random.default_rng(random_state)
    model, explain, background, fold = _resolve(
        source, X, fold, n_explain, n_background, rng
    )
    names = list(explain.columns)

    try:
        explanation = shap.TreeExplainer(model, background)(explain)
        kind = "tree"
    except Exception:  # shap raises several types for unsupported models
        evals = max_evals or 2 * len(names) + 1
        cost = len(explain) * evals * len(background)
        warnings.warn(
            f"{type(model).__name__} is not a tree model shap supports; "
            f"using the permutation explainer (~{cost:,} predicted rows).",
            stacklevel=2,
        )

        def predict(data: np.ndarray) -> np.ndarray:
            return model.predict(pd.DataFrame(data, columns=names))

        explainer = shap.PermutationExplainer(
            predict,
            shap.maskers.Independent(background, max_samples=len(background)),
            feature_names=names,
            seed=random_state,
        )
        explanation = explainer(explain, max_evals=evals, silent=True)
        kind = "permutation"

    return {
        "explanation": explanation,
        "features": explain,
        "explainer": kind,
        "fold": fold,
    }


# ---------------------------------------------------------------------------
# 2. Plots
# ---------------------------------------------------------------------------
def plot_shap(
    explanation: shap.Explanation,
    *,
    row: int = 0,
    max_display: int = 15,
    figsize: tuple[float, float] = (10, 6),
    show: bool = True,
) -> dict:
    """Bar, beeswarm and waterfall plots of ``explanation``.

    Parameters
    ----------
    explanation : the ``explanation`` returned by ``compute_shap``.
    row : the observation the waterfall plot breaks down.
    max_display : features shown before the rest are grouped.
    figsize : size of each figure.
    show : display the figures. With ``False`` they are closed, so a
        notebook does not draw them, and only returned.

    Returns
    -------
    dict
        ``bar``, ``beeswarm`` and ``waterfall``, each a Figure.
    """
    figs: dict[str, Any] = {}

    fig, ax = plt.subplots(figsize=figsize)
    shap.plots.bar(explanation, max_display=max_display, ax=ax, show=False)
    ax.set_title("Importancia media (|SHAP|)", fontweight="bold")
    ax.set_xlabel(f"Media de |SHAP| ({AXIS_LABEL.lower()})")
    figs["bar"] = fig

    fig, ax = plt.subplots(figsize=figsize)
    shap.plots.beeswarm(
        explanation,
        max_display=max_display,
        ax=ax,
        show=False,
        plot_size=None,
        color_bar_label="Valor de la variable",
    )
    ax.set_title("Impacto por observación", fontweight="bold")
    ax.set_xlabel(f"SHAP ({AXIS_LABEL.lower()})")
    figs["beeswarm"] = fig

    fig = plt.figure(figsize=figsize)
    shap.plots.waterfall(explanation[row], max_display=max_display, show=False)
    fig = plt.gcf()
    fig.suptitle(
        f"Descomposición de la predicción de la fila {row}", fontweight="bold"
    )
    figs["waterfall"] = fig

    if show:
        plt.show()
    else:
        for fig in figs.values():
            plt.close(fig)
    return figs


# ---------------------------------------------------------------------------
# 3. Orchestrator
# ---------------------------------------------------------------------------
def shap_report(
    source: Any,
    X: pd.DataFrame,
    *,
    fold: int = -1,
    n_explain: int = 200,
    n_background: int = 100,
    max_evals: int | None = None,
    row: int = 0,
    max_display: int = 15,
    figsize: tuple[float, float] = (10, 6),
    random_state: int = 0,
    show: bool = True,
) -> dict:
    """``compute_shap`` then ``plot_shap``; see both for the parameters.

    Returns
    -------
    dict
        What ``compute_shap`` returns, plus ``figs`` from ``plot_shap``.
    """
    result = compute_shap(
        source,
        X,
        fold=fold,
        n_explain=n_explain,
        n_background=n_background,
        max_evals=max_evals,
        random_state=random_state,
    )
    result["figs"] = plot_shap(
        result["explanation"],
        row=row,
        max_display=max_display,
        figsize=figsize,
        show=show,
    )
    return result
