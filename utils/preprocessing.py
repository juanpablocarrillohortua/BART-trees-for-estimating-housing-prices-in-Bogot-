"""sklearn building blocks for the listing-price pipeline.

The preparation of ``notebooks/data_cleaning.ipynb`` and
``notebooks/data_preparation.ipynb``, reshaped so it can be learned inside
each cross-validation fold. The stateless steps are plain functions meant
for ``FunctionTransformer``; the three classes cover what stock sklearn has
no object for -- imputing a category from its nearest neighbour, frequency
encoding, and dropping outlier rows while fitting.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import (
    BaseEstimator,
    RegressorMixin,
    TransformerMixin,
    clone,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.validation import check_is_fitted

#: Output column -> the amenity name it flags (data_preparation cells 7-8).
AMENITY_FLAGS: dict[str, str] = {
    "Trans Publico cerc": "Trans. Público cercano",
    "Zona Residencial": "Zona Residencial",
}


# ---------------------------------------------------------------------------
# Stateless steps, for FunctionTransformer
# ---------------------------------------------------------------------------
def impute_bimodal(
    X: pd.DataFrame, values: Sequence[float], seed: int = 42
) -> pd.DataFrame:
    """Fill NaN with an even random draw among ``values``.

    Spreads the missing rows over the two most common levels instead of
    piling them on one, so the mode is not inflated. The same ``seed`` is
    drawn on every call, which keeps the step stateless.
    """
    rng = np.random.default_rng(seed)
    out = pd.DataFrame(X).copy()
    for col in out.columns:
        missing = out[col].isna()
        out.loc[missing, col] = rng.choice(values, size=int(missing.sum()))
    return out


def clip_round(X: pd.DataFrame) -> pd.DataFrame:
    """Clip at 0 and round to int: MICE returns counts as fractions."""
    return pd.DataFrame(X).clip(lower=0).round().astype(int)


def months_listed(X: pd.DataFrame, anchor: str = "2026-08") -> pd.DataFrame:
    """Months between each publication date and ``anchor``.

    ``anchor`` is the scrape month, fixed rather than learned, so a fold
    with no recent listings measures from the same point as the others.
    """
    published = pd.to_datetime(pd.DataFrame(X).iloc[:, 0])
    end = pd.Period(anchor, freq="M")
    months = (end.year - published.dt.year) * 12 + (
        end.month - published.dt.month
    )
    return months.astype(int).rename("meses_ofertado").to_frame()


def months_listed_names(
    transformer: Any, input_features: Sequence[str]
) -> np.ndarray:
    """``feature_names_out`` for :func:`months_listed`."""
    return np.asarray(["meses_ofertado"], dtype=object)


def amenity_features(X: pd.DataFrame) -> pd.DataFrame:
    """Two amenity flags and the amenity count, from the pipe-joined list.

    ``n_amenities`` is 0 for "Ninguna", the value the imputer gives a
    listing that publishes no amenities.
    """
    lists = pd.DataFrame(X).iloc[:, 0].str.split("|")
    out = pd.DataFrame(index=lists.index)
    for col, name in AMENITY_FLAGS.items():
        out[col] = lists.apply(lambda x, name=name: name in x).astype(int)
    out["n_amenities"] = lists.apply(
        lambda x: len(x) if "Ninguna" not in x else 0
    )
    return out


def amenity_feature_names(
    transformer: Any, input_features: Sequence[str]
) -> np.ndarray:
    """``feature_names_out`` for :func:`amenity_features`."""
    return np.asarray([*AMENITY_FLAGS, "n_amenities"], dtype=object)


def log_feature_names(
    transformer: Any, input_features: Sequence[str]
) -> np.ndarray:
    """``feature_names_out`` for a ``np.log`` step: prefixes ``log_``."""
    return np.asarray([f"log_{c}" for c in input_features], dtype=object)


def grouped_feature_name(input_feature: str, category: Any) -> str:
    """``OneHotEncoder`` ``feature_name_combiner`` for a grouped column.

    Gives ``<feature>_grouped_<category>``, and ``..._grouped_Otros`` for
    the pooled infrequent categories -- the names data_preparation used.
    """
    label = "Otros" if category == "infrequent_sklearn" else category
    return f"{input_feature}_grouped_{label}"


# ---------------------------------------------------------------------------
# Transformers
# ---------------------------------------------------------------------------
class NearestNeighborImputer(TransformerMixin, BaseEstimator):
    """Fill a categorical column from its nearest neighbours in space.

    Takes the column to fill first and the coordinate columns after it;
    returns the filled column alone, so the coordinates must also be passed
    through by another branch of the ``ColumnTransformer``.

    Parameters
    ----------
    n_neighbors : int, default=1
        Neighbours consulted, weighted by inverse distance.
    """

    def __init__(self, n_neighbors: int = 1) -> None:
        self.n_neighbors = n_neighbors

    def fit(self, X: pd.DataFrame, y: Any = None) -> NearestNeighborImputer:
        X = pd.DataFrame(X)
        target, coords = X.iloc[:, 0], X.iloc[:, 1:]
        known = target.notna() & coords.notna().all(axis=1)
        self.knn_ = KNeighborsClassifier(
            n_neighbors=self.n_neighbors, weights="distance"
        ).fit(coords[known].to_numpy(), target[known].to_numpy())
        self.feature_names_in_ = X.columns.to_numpy(dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "knn_")
        X = pd.DataFrame(X)
        target, coords = X.iloc[:, 0].copy(), X.iloc[:, 1:]
        missing = target.isna() & coords.notna().all(axis=1)
        if missing.any():
            target[missing] = self.knn_.predict(coords[missing].to_numpy())
        return target.to_frame()

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        check_is_fitted(self, "knn_")
        return self.feature_names_in_[:1]


class FrequencyEncoder(TransformerMixin, BaseEstimator):
    """Replace each category by its share of the training rows.

    Parameters
    ----------
    top_n : int or None, default=None
        Keep only the ``top_n`` most frequent categories and pool the rest
        under ``other`` before counting. ``None`` keeps them all.
    other : str, default="Otros"
        Name of the pooled group. A category unseen in fit is pooled here
        too; with ``top_n=None`` there is no such group and it gets 0.
    suffix : str, default="_frec"
        Appended to each input name to form the output name.
    """

    def __init__(
        self,
        top_n: int | None = None,
        other: str = "Otros",
        suffix: str = "_frec",
    ) -> None:
        self.top_n = top_n
        self.other = other
        self.suffix = suffix

    def fit(self, X: pd.DataFrame, y: Any = None) -> FrequencyEncoder:
        X = pd.DataFrame(X)
        self.kept_: dict[str, pd.Index] = {}
        self.frequencies_: dict[str, pd.Series] = {}
        for col in X.columns:
            counts = X[col].value_counts()
            if self.top_n is not None:
                counts = counts.nlargest(self.top_n)
            self.kept_[col] = counts.index
            self.frequencies_[col] = self._pool(X[col]).value_counts(
                normalize=True
            )
        self.feature_names_in_ = X.columns.to_numpy(dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def _pool(self, s: pd.Series) -> pd.Series:
        return s.where(s.isin(self.kept_[s.name]), self.other)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "frequencies_")
        X = pd.DataFrame(X)
        return pd.DataFrame(
            {
                f"{col}{self.suffix}": self._pool(X[col])
                .map(self.frequencies_[col])
                .astype(float)
                .fillna(0.0)
                for col in X.columns
            },
            index=X.index,
        )

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        check_is_fitted(self, "frequencies_")
        return np.asarray(
            [f"{col}{self.suffix}" for col in self.feature_names_in_],
            dtype=object,
        )


# ---------------------------------------------------------------------------
# Row filter
# ---------------------------------------------------------------------------
class OutlierFilterRegressor(RegressorMixin, BaseEstimator):
    """Fit ``regressor`` only on the rows that pass the outlier rules.

    sklearn steps cannot drop rows, so this filter wraps the steps that
    come after it instead. ``fit`` drops rows from X and y, then fits a
    clone of ``regressor`` on what is left; ``predict`` passes every row
    through. Test rows are never dropped -- the model just never learned
    from the outliers.

    The rules run in the order of ``data_cleaning.ipynb``: business rules
    first, then IQR fences computed on the rows those rules kept.

    Parameters
    ----------
    regressor : estimator
        Everything downstream of the filter.
    bounds : mapping of str to (low, high, inclusive), default=None
        Business rules, applied with ``Series.between``.
    iqr_columns : sequence of str, default=()
        Columns whose ``iqr_multiplier`` x IQR fences are applied next.
    ratio_columns : mapping of str to (numerator, denominator), default=None
        Columns derived before the rules run, e.g. a price per m2 the
        regressor itself must not see.
    iqr_multiplier : float, default=3.0
    target_name : str, default="price_amount"
        Name under which y is visible to ``bounds``, ``iqr_columns`` and
        ``ratio_columns``. The rules see y as given, so a log transform of
        the target belongs inside ``regressor``.

    Attributes
    ----------
    regressor_ : estimator
        The clone fitted on the kept rows.
    fences_ : DataFrame
        ``lower`` and ``upper`` IQR fence per column.
    kept_ : ndarray of bool
        Which training rows the regressor saw.
    n_dropped_business_, n_dropped_iqr_ : int
        Rows each stage removed.
    """

    def __init__(
        self,
        regressor: Any,
        bounds: Mapping[str, tuple[float, float, str]] | None = None,
        iqr_columns: Sequence[str] = (),
        ratio_columns: Mapping[str, tuple[str, str]] | None = None,
        iqr_multiplier: float = 3.0,
        target_name: str = "price_amount",
    ) -> None:
        self.regressor = regressor
        self.bounds = bounds
        self.iqr_columns = iqr_columns
        self.ratio_columns = ratio_columns
        self.iqr_multiplier = iqr_multiplier
        self.target_name = target_name

    def fit(self, X: pd.DataFrame, y: Any) -> OutlierFilterRegressor:
        frame = X.assign(**{self.target_name: np.asarray(y)})
        for name, (num, den) in (self.ratio_columns or {}).items():
            frame[name] = frame[num] / frame[den]

        passed = pd.Series(True, index=frame.index)
        for col, (low, high, inclusive) in (self.bounds or {}).items():
            passed &= frame[col].between(low, high, inclusive=inclusive)

        cols = list(self.iqr_columns)
        q1 = frame.loc[passed, cols].quantile(0.25)
        q3 = frame.loc[passed, cols].quantile(0.75)
        spread = self.iqr_multiplier * (q3 - q1)
        self.fences_ = pd.DataFrame(
            {"lower": q1 - spread, "upper": q3 + spread}
        )
        within = (
            frame[cols].ge(self.fences_["lower"])
            & frame[cols].le(self.fences_["upper"])
        ).all(axis=1)

        kept = (passed & within).to_numpy()
        self.n_dropped_business_ = int((~passed).sum())
        self.n_dropped_iqr_ = int((passed & ~within).sum())
        self.kept_ = kept
        y_kept = y.iloc[kept] if hasattr(y, "iloc") else np.asarray(y)[kept]
        self.regressor_ = clone(self.regressor).fit(X.iloc[kept], y_kept)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "regressor_")
        return self.regressor_.predict(X)
