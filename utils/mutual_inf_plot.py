"""
Análisis de relevancia de variables predictoras frente a un target binario
o continuo.

Combina dos métricas:
  - Correlación (valor absoluto) con el target.
  - Información mutua (mutual_info_classif para un target categórico,
    mutual_info_regression para uno continuo).

Ambas se normalizan con MinMax y se promedian para obtener un score 0–1.

El target se extrae directamente del DataFrame: basta con pasar `df` y `target`
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.feature_selection import (
    mutual_info_classif,
    mutual_info_regression,
)

SUBTITULO = "Promedio normalizado (MinMax) de Correlación e Información Mutua"

#: Hasta cuántos valores distintos un target entero se trata como clases.
MAX_CLASES = 20


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
def _preparar(
    df: pd.DataFrame,
    target: str,
    predictoras: Optional[Iterable[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Valida el target, separa X e y y elimina filas con nulos.

    Returns
    -------
    (X, y, predictoras)
    """
    if target not in df.columns:
        raise KeyError(
            f"La columna target '{target}' no está en el DataFrame. "
            f"Columnas disponibles: {list(df.columns)}"
        )

    predictoras = (
        list(predictoras)
        if predictoras is not None
        else df.columns.difference([target]).tolist()
    )
    if not predictoras:
        raise ValueError("No hay columnas predictoras además del target.")

    datos = df[predictoras + [target]].dropna()
    if datos.empty:
        raise ValueError("No quedan filas tras eliminar nulos.")

    return datos[predictoras], datos[target], predictoras


def _escalar(serie: pd.Series) -> pd.Series:
    """Escala una serie al rango [0, 1] conservando el índice.

    A mano y no con ``MinMaxScaler``: con
    ``set_config(transform_output="pandas")`` el escalador devuelve un
    DataFrame. Una serie constante queda en 0.
    """
    rango = serie.max() - serie.min()
    if rango == 0:
        return pd.Series(0.0, index=serie.index)
    return (serie - serie.min()) / rango


def _es_clasificacion(y: pd.Series, tipo: str) -> bool:
    """Decide si la información mutua se calcula como clasificación."""
    if tipo == "clasificacion":
        return True
    if tipo == "regresion":
        return False
    if tipo != "auto":
        raise ValueError(
            "`tipo` debe ser 'auto', 'clasificacion' o 'regresion'."
        )
    if not pd.api.types.is_numeric_dtype(y) or pd.api.types.is_bool_dtype(y):
        return True
    return pd.api.types.is_integer_dtype(y) and y.nunique() <= MAX_CLASES


# --------------------------------------------------------------------------- #
# 1. Métricas
# --------------------------------------------------------------------------- #
def calcular_informacion_mutua(
    df: pd.DataFrame,
    target: str = "is_popular",
    predictoras: Optional[Iterable[str]] = None,
    discretas: bool | Sequence[bool] = False,
    random_state: int = 42,
    tipo: str = "auto",
) -> pd.Series:
    """
    Información mutua entre cada predictora y el target.

    Parameters
    ----------
    tipo : 'auto', 'clasificacion' o 'regresion'. 'auto' usa clasificación
        si el target es booleano, categórico o entero con pocos valores
        (``MAX_CLASES``), y regresión en otro caso.

    Returns
    -------
    pd.Series indexada por nombre de variable, ordenada ascendentemente.
    """
    X, y, predictoras = _preparar(df, target, predictoras)  # noqa: N806

    mutual_info = (
        mutual_info_classif
        if _es_clasificacion(y, tipo)
        else mutual_info_regression
    )
    scores = mutual_info(
        X.to_numpy(),
        y,
        discrete_features=discretas,
        random_state=random_state,
    )
    return pd.Series(
        scores, index=predictoras, name="info_mutua"
    ).sort_values()


def calcular_correlacion(
    df: pd.DataFrame,
    target: str = "is_popular",
    predictoras: Optional[Iterable[str]] = None,
    metodo: str = "pearson",
) -> pd.Series:
    """Correlación de cada predictora con el target (con signo)."""
    X, y, predictoras = _preparar(df, target, predictoras)  # noqa: N806

    corr = X.join(y).corr(method=metodo, numeric_only=True)[target]
    return corr.drop(target).reindex(predictoras).rename("correlacion")


def calcular_score_relevancia(
    df: pd.DataFrame,
    target: str = "is_popular",
    predictoras: Optional[Iterable[str]] = None,
    pesos: Tuple[float, float] = (0.5, 0.5),
    random_state: int = 42,
    tipo: str = "auto",
) -> pd.DataFrame:
    """
    Score combinado = promedio ponderado de |correlación| y MI, ambas
    normalizadas.

    Parameters
    ----------
    pesos : (peso_correlacion, peso_mi). Se normalizan para sumar 1.
    tipo : tipo de información mutua; ver ``calcular_informacion_mutua``.

    Returns
    -------
    DataFrame indexado por variable con columnas:
    ['correlacion', 'info_mutua', 'corr_scaled', 'mi_scaled', 'score'],
    ordenado por score ascendente (listo para barh).
    """
    _, _, predictoras = _preparar(df, target, predictoras)

    corr = calcular_correlacion(df, target, predictoras)
    mi = calcular_informacion_mutua(
        df, target, predictoras, random_state=random_state, tipo=tipo
    ).reindex(predictoras)

    if sum(pesos) == 0:
        raise ValueError("La suma de `pesos` no puede ser 0.")
    w_corr, w_mi = (p / sum(pesos) for p in pesos)

    df_score = pd.DataFrame(
        {
            "correlacion": corr,
            "info_mutua": mi,
            "corr_scaled": _escalar(corr.abs()),
            "mi_scaled": _escalar(mi),
        }
    )
    df_score["score"] = (
        w_corr * df_score["corr_scaled"] + w_mi * df_score["mi_scaled"]
    )
    df_score.index.name = "variable"

    return df_score.sort_values("score")


# --------------------------------------------------------------------------- #
# 2. Visualización
# --------------------------------------------------------------------------- #
def graficar_score_relevancia(
    df_score: pd.DataFrame,
    columna: str = "score",
    titulo: str = "Score de Relevancia Combinado",
    subtitulo: str = SUBTITULO,
    top_n: Optional[int] = None,
    color: str = "#7b51d3",
    figsize: Tuple[float, float] = (10, 8),
    mostrar: bool = True,
) -> Tuple[Figure, Axes]:
    """
    Barras horizontales con estilo editorial para el score de relevancia.

    Parameters
    ----------
    top_n : si se indica, grafica solo las `top_n` variables con mayor score.
    """
    datos = df_score.sort_values(columna)
    if top_n is not None:
        datos = datos.tail(top_n)

    fig, ax = plt.subplots(figsize=figsize)

    fig.text(
        0.1, 0.95, titulo, fontsize=22, fontweight="bold", color="#2c3e50"
    )
    fig.text(0.1, 0.92, subtitulo, fontsize=13, color="#7f8c8d")

    barras = ax.barh(
        datos.index, datos[columna], color=color, edgecolor="white", height=0.6
    )

    for barra, valor in zip(barras, datos[columna]):
        ax.text(
            valor + 0.008,
            barra.get_y() + barra.get_height() / 2,
            f"{valor:.3f}",
            va="center",
            ha="left",
            fontsize=9,
            color="#333333",
        )

    ax.set_xlabel("Score (0 – 1)", fontsize=11)
    ax.set_xlim(0, 1.12)
    ax.grid(axis="x", alpha=0.3)
    sns.despine(left=True, ax=ax)
    fig.subplots_adjust(top=0.90, left=0.18, right=0.95, bottom=0.10)

    if mostrar:
        plt.show()

    return fig, ax


# --------------------------------------------------------------------------- #
# 3. Orquestador
# --------------------------------------------------------------------------- #
def analizar_relevancia_caracteristicas(
    df: pd.DataFrame,
    target: str = "is_popular",
    predictoras: Optional[Iterable[str]] = None,
    pesos: Tuple[float, float] = (0.5, 0.5),
    top_n: Optional[int] = None,
    graficar: bool = True,
    random_state: int = 42,
    tipo: str = "auto",
) -> pd.DataFrame:
    """Calcula el score de relevancia y opcionalmente lo grafica."""
    df_score = calcular_score_relevancia(
        df,
        target=target,
        predictoras=predictoras,
        pesos=pesos,
        random_state=random_state,
        tipo=tipo,
    )

    if graficar:
        graficar_score_relevancia(df_score, top_n=top_n)

    return df_score


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Ejemplo de uso:
    # df_score = analizar_relevancia(df_final, target="is_popular", top_n=15)
    # print(df_score.sort_values("score", ascending=False).head(10))
    pass
