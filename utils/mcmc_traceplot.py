"""MCMC trace plots of a fitted BART, or of one fold of a CV result.

``traceplot`` takes either the dict returned by the notebooks'
``cross_validate_model_cached`` (its last fold by default) or a fitted
model, pulls the posterior chains of sigma^2 and of f(x) for a few random
rows, and draws one row per quantity: the density of each chain on the
left and its trace on the right. ``extract_chains`` and ``plot_trace`` are
the two halves, so the slow part (loading a fold) can be cached on its own.

f(x) is on the scale the model was fitted on: log(price) in the project's
pipelines, which fit the model inside a ``TransformedTargetRegressor`` with
``np.log``.
"""

from __future__ import annotations

import gc
from collections.abc import Mapping
from typing import Any

import arviz as az
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import config_context

# --- STYLE PALETTE (the one triple_plot uses) ---
C_DARK = "#2c3e50"
C_MID = "#34495e"
C_SOFT = "#7f8c8d"
C_GRID = "#eeeeee"
C_ORANGE = "#e67e22"
CHAIN_COLORS = ("#5dade2", C_ORANGE)
C_OTHER_CHAIN = "#95a5a6"

#: R-hat above this is flagged, the threshold ``mcmc_diagnostics`` uses.
RHAT_THRESHOLD = 1.01


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
    """The bare model of a fitted pipeline and ``rows`` as it sees them."""
    inner = pipeline.named_steps["outliers"].regressor_.regressor_
    with config_context(transform_output="pandas"):
        imputed = pipeline.named_steps["imputation"].transform(rows)
        features = inner[:-1].transform(imputed)
    return inner[-1], features


def _check_has_chains(model: Any) -> None:
    if not hasattr(model, "posterior_draws"):
        raise TypeError(
            f"{type(model).__name__} has no posterior chains to trace; "
            "only a BART with posterior_draws (BARTAtrRegressor) does."
        )


def _resolve(
    source: Any,
    X: pd.DataFrame,
    fold: int,
    n_rows: int,
    rng: np.random.Generator,
) -> tuple[Any, pd.DataFrame, list, int | None]:
    """Return (model, the sampled rows as it sees them, labels, fold)."""
    if _is_cv_result(source):
        fold = range(len(source["estimator"]))[fold]  # -1 -> last fold
        positions = np.asarray(source["test_indices"][fold])
    else:
        positions = np.arange(len(X))
    chosen = np.sort(
        rng.choice(positions, size=min(n_rows, len(positions)), replace=False)
    )
    rows = X.iloc[chosen]
    labels = list(rows.index)

    if _is_cv_result(source):
        pipeline = joblib.load(source["estimator"][fold])
        model, features = _model_and_features(pipeline, rows)
        del pipeline  # ~2.6 GB of MICE trees; only the model is needed
        gc.collect()
        _check_has_chains(model)
        return model, features, labels, fold

    if _is_project_pipeline(source):
        model, features = _model_and_features(source, rows)
        _check_has_chains(model)
        return model, features, labels, None
    _check_has_chains(source)
    return source, rows, labels, None


# ---------------------------------------------------------------------------
# 1. Chains
# ---------------------------------------------------------------------------
def extract_chains(
    source: Any,
    X: pd.DataFrame,
    *,
    fold: int = -1,
    n_rows: int = 3,
    random_state: int = 0,
) -> dict:
    """Posterior chains of sigma^2 and of f(x) for ``n_rows`` random rows.

    Parameters
    ----------
    source : one of
        - the dict returned by ``cross_validate_model_cached`` (it needs
          ``estimator``, ``test_indices`` and ``train_indices``). Fold
          ``fold`` is loaded from disk and the rows are drawn from its test
          rows. The pipeline is freed once the chains are taken out.
        - a fitted pipeline of the project's shape (``imputation`` then
          ``outliers``): ``X`` is raw rows, transformed like the model saw
          them.
        - a fitted ``BARTAtrRegressor``: ``X`` is already its feature
          matrix.
        Anything without ``posterior_draws`` raises ``TypeError``.
    X : the rows. For a CV result, the same ``X`` that was cross-validated,
        since the fold indices are positions into it.
    fold : which CV fold; ``-1`` is the last one.
    n_rows : how many rows get their f(x) traced.
    random_state : seed for the row sample.

    Returns
    -------
    dict
        ``sigma2`` (chain, draw), ``f`` (chain, draw, row), ``rows`` (the
        ``X`` index labels of the traced rows), ``fold`` (``None`` unless
        ``source`` is a CV result) and ``diagnostics``, a DataFrame with
        ``rhat`` and ``ess_bulk`` per traced quantity, computed by arviz on
        every draw.
    """
    rng = np.random.default_rng(random_state)
    model, features, labels, fold = _resolve(source, X, fold, n_rows, rng)
    f = model.posterior_draws(features)  # (chain, draw, row)
    sigma2 = np.asarray(model.model_.global_var_samples).reshape(
        f.shape[0], -1
    )
    del model, features
    gc.collect()

    idata = az.from_dict({"posterior": {"sigma2": sigma2, "f": f}})
    rhat = az.rhat(idata)
    ess = az.ess(idata, method="bulk")
    diagnostics = pd.DataFrame(
        {
            "rhat": [
                float(rhat["sigma2"].values),
                *np.ravel(rhat["f"].values),
            ],
            "ess_bulk": [
                float(ess["sigma2"].values),
                *np.ravel(ess["f"].values),
            ],
        },
        index=pd.Index(
            ["sigma2", *[f"f[{label}]" for label in labels]],
            name="quantity",
        ),
    )
    return {
        "sigma2": sigma2,
        "f": f,
        "rows": labels,
        "fold": fold,
        "diagnostics": diagnostics,
    }


# ---------------------------------------------------------------------------
# 2. Plot
# ---------------------------------------------------------------------------
def _chain_color(i: int) -> str:
    return CHAIN_COLORS[i] if i < len(CHAIN_COLORS) else C_OTHER_CHAIN


def _draw_row(
    ax_kde: plt.Axes,
    ax_trace: plt.Axes,
    series: np.ndarray,
    label: str,
    rhat: float,
    ess: float,
    max_points: int,
) -> None:
    """Density and trace of one quantity; ``series`` is (chain, draw)."""
    n_draws = series.shape[1]
    step = max(1, int(np.ceil(n_draws / max_points)))
    iterations = np.arange(n_draws)[::step]
    grid = np.linspace(series.min(), series.max(), 256)

    for i, chain in enumerate(series):
        color = _chain_color(i)
        if np.ptp(chain) > 0:  # a constant chain has no density
            ax_kde.plot(
                grid, stats.gaussian_kde(chain)(grid), color=color, lw=1.6
            )
        else:
            ax_kde.axvline(chain[0], color=color, lw=1.6)
        ax_trace.plot(
            iterations,
            chain[::step],
            color=color,
            lw=0.6,
            alpha=0.7,
            label=f"Cadena {i + 1}",
        )

    flagged = not np.isfinite(rhat) or rhat > RHAT_THRESHOLD
    ax_kde.set_title(
        label, loc="left", fontsize=12, fontweight="bold", color=C_MID
    )
    ax_trace.set_title(
        f"R̂ = {rhat:.3f} · ESS = {ess:,.0f}",
        loc="right",
        fontsize=11,
        fontweight="bold" if flagged else "normal",
        color=C_ORANGE if flagged else C_SOFT,
    )
    ax_kde.set_ylabel("Densidad", fontsize=10)
    ax_trace.set_xlim(0, n_draws - 1)
    for ax in (ax_kde, ax_trace):
        ax.grid(color=C_GRID, linewidth=1)
        ax.tick_params(colors=C_SOFT, labelsize=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)


def plot_trace(
    chains: dict,
    *,
    max_points: int = 2000,
    figsize: tuple[float, float] | None = None,
    show: bool = True,
) -> plt.Figure:
    """Density and trace per chain of every quantity in ``chains``.

    Parameters
    ----------
    chains : the dict returned by ``extract_chains``.
    max_points : draws drawn per chain on a trace; beyond it every k-th
        draw is drawn. The densities and the R-hat/ESS use every draw.
    figsize : size of the figure; ``None`` scales it with the rows.
    show : display the figure. With ``False`` it is closed, so a notebook
        does not draw it, and only returned.

    Returns
    -------
    Figure
        One row per quantity (sigma^2 first, then f(x) per row), density
        on the left and trace on the right. R-hat above 1.01 is orange.
    """
    sigma2, f = chains["sigma2"], chains["f"]
    diag = chains["diagnostics"]
    n_quantities = 1 + f.shape[2]
    if figsize is None:
        figsize = (12, 2.4 * n_quantities)

    with plt.rc_context():
        fig, axes = plt.subplots(
            n_quantities,
            2,
            figsize=figsize,
            gridspec_kw={"width_ratios": [1, 2.5]},
            squeeze=False,
        )
        quantities = [("σ²", sigma2)] + [
            (f"f(x) · fila {label}  [log(precio)]", f[:, :, j])
            for j, label in enumerate(chains["rows"])
        ]
        for (label, series), (ax_kde, ax_trace), (_, d) in zip(
            quantities, axes, diag.iterrows(), strict=True
        ):
            _draw_row(
                ax_kde,
                ax_trace,
                series,
                label,
                d["rhat"],
                d["ess_bulk"],
                max_points,
            )
        axes[-1, 1].set_xlabel("Iteración", fontsize=10)
        legend = axes[0, 1].legend(
            frameon=False, fontsize=9, loc="upper right"
        )
        for line in legend.get_lines():
            line.set_linewidth(2)

        title = "Traceplot MCMC"
        if chains["fold"] is not None:
            title += f" · pliegue {chains['fold']}"
        fig.suptitle(
            title,
            x=0.01,
            ha="left",
            fontsize=16,
            fontweight="bold",
            color=C_DARK,
        )
        fig.tight_layout()
        if show:
            plt.show()
        else:
            plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# 3. Orchestrator
# ---------------------------------------------------------------------------
def traceplot(
    source: Any,
    X: pd.DataFrame,
    *,
    fold: int = -1,
    n_rows: int = 3,
    max_points: int = 2000,
    figsize: tuple[float, float] | None = None,
    random_state: int = 0,
    show: bool = True,
) -> dict:
    """``extract_chains`` then ``plot_trace``; see both for the parameters.

    Returns
    -------
    dict
        What ``extract_chains`` returns, plus ``fig`` from ``plot_trace``.
    """
    result = extract_chains(
        source, X, fold=fold, n_rows=n_rows, random_state=random_state
    )
    result["fig"] = plot_trace(
        result, max_points=max_points, figsize=figsize, show=show
    )
    return result
