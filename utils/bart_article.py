"""
BART(Atr) como estimador de scikit-learn (Cao, Wu & Wang, 2024).

Permite usarlo dentro de Pipeline, GridSearchCV, cross_validate, etc.

Requiere:  pip install stochtree scikit-learn numpy scipy pandas
"""

from __future__ import annotations

from typing import Any, Literal, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import chi2, norm
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted, validate_data
from stochtree import BARTModel


# ---------------------------------------------------------------------------
# Piezas matemáticas del recorte (ecuaciones 4, 6, 7 y 8 del paper)
# ---------------------------------------------------------------------------
def folded_normal_quantiles(n: int) -> NDArray[np.float64]:
    """Cuantiles teóricos de la normal plegada estándar (ec. 4).

    Parameters
    ----------
    n : int
        Número de residuos.

    Returns
    -------
    z : ndarray de forma (n,)
        z_j = Phi^{-1}((1 + p_j)/2), con p_j = j/(n+1). Es dónde debería caer
        el j-ésimo residuo absoluto ordenado si el ruido fuera N(0, 1).
    """
    p_tilde: NDArray[np.float64] = np.arange(1, n + 1) / (n + 1)
    return norm.ppf((1 + p_tilde) / 2)


def robust_sigma2(
    abs_sorted: NDArray[np.float64], z: NDArray[np.float64], frac: float = 0.25
) -> float:
    """Estimación robusta de sigma^2 de los datos típicos (ec. 6).

    Parameters
    ----------
    abs_sorted : ndarray de forma (n,)
        Residuos absolutos ordenados de menor a mayor.
    z : ndarray de forma (n,)
        Cuantiles de `folded_normal_quantiles`.
    frac : float, default=0.25
        Fracción de residuos más pequeños que se usa.

    Returns
    -------
    sigma2 : float
        Suma observada de residuos^2 / suma esperada bajo N(0, 1).
    """
    k: int = max(1, int(np.floor(len(abs_sorted) * frac)))
    return float(np.sum(abs_sorted[:k] ** 2) / np.sum(z[:k] ** 2))


def estimate_p(
    abs_sorted: NDArray[np.float64],
    z: NDArray[np.float64],
    sigma2: float,
    Q: float,  # noqa: N803 - the paper's name
) -> tuple[float, NDArray[np.float64]]:
    """Proporción estimada de datos limpios (ec. 7).

    Parameters
    ----------
    abs_sorted : ndarray de forma (n,)
        Residuos absolutos ordenados de menor a mayor.
    z : ndarray de forma (n,)
        Cuantiles de `folded_normal_quantiles`.
    sigma2 : float
        Varianza robusta de `robust_sigma2`.
    Q : float
        Tolerancia del cociente r_i.

    Returns
    -------
    p_hat : float
        |S| / n, con S = {i : s_i^2 / E(s_i^2) <= Q}.
    ratio : ndarray de forma (n,)
        Los cocientes r_i = s_i^2 / E(s_i^2) de cada posición.
    """
    i: NDArray[np.int64] = np.arange(1, len(abs_sorted) + 1)
    ratio: NDArray[np.float64] = (np.cumsum(abs_sorted**2) / i) / (
        sigma2 * np.cumsum(z**2) / i
    )
    return float(np.sum(ratio <= Q) / len(abs_sorted)), ratio


# ---------------------------------------------------------------------------
# Estimador
# ---------------------------------------------------------------------------
class BARTAtrRegressor(RegressorMixin, BaseEstimator):
    """BART con recorte adaptativo de outliers (BART(Atr)).

    Ajusta BART, calcula los residuos de todos los puntos de entrenamiento,
    estima la proporción de datos limpios p_hat comparando los residuos
    ordenados con los de una normal, y reentrena BART sin los puntos cuyo
    residuo supera el umbral. Repite hasta que p_hat se estabiliza.

    Parameters
    ----------
    Recorte (capa de BART(Atr))
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Q : float, default=1.28
        Tolerancia del cociente r_i = s_i^2 / E(s_i^2). Una posición i se
        considera limpia si r_i <= Q. Mayor Q recorta menos; menor Q recorta
        más. Valor elegido por los autores en sus simulaciones.
    trim : bool, default=True
        Si es True corre BART(Atr). Si es False solo ajusta el BART inicial
        con todos los datos, es decir BART(Def); útil como línea base.
    sigma_frac : float, default=0.25
        Fracción de residuos más pequeños usada para estimar sigma^2 (ec. 6).
        Supone que al menos esa fracción de los datos es limpia.
    p_min : float, default=0.5
        Proporción mínima de datos que siempre se conserva. No está en el
        paper: evita que una iteración inestable recorte casi todo.
        Con p_min=0 el método es el original.
    tol : float, default=1e-4
        Criterio de parada: |p_hat(t) - p_hat(t-1)| / p_hat(t) < tol.
    max_iter : int, default=20
        Máximo de reentrenamientos.

    BART (árboles y priors, Chipman et al. 2010; Tabla 1 del paper)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    num_trees : int, default=50
        Número de árboles m del ensamble.
    alpha : float, default=0.95
        Probabilidad base de que un nodo se divida. La probabilidad de
        dividir un nodo a profundidad d es alpha * (1 + d)^(-beta).
    beta : float, default=2.0
        Penalización de la profundidad: mayor beta da árboles más pequeños.
    k : float, default=2.0
        Encogimiento de los valores de las hojas: sigma_mu = rango(y) /
        (2 k sqrt(m)). Mayor k da predicciones más suaves y conservadoras.
    nu : float, default=3.0
        Grados de libertad del prior sigma^2 ~ nu * lambda / chi2_nu.
        Mayor nu concentra el prior.
    q : float, default=0.9
        Calibra lambda: probabilidad a priori de que sigma sea menor que la
        desviación de los residuos de una regresión lineal (OLS).

    MCMC y ejecución
    ~~~~~~~~~~~~~~~~
    num_burnin : int, default=500
        Iteraciones iniciales de cada cadena que se descartan.
    num_mcmc : int, default=1000
        Draws posteriores que se guardan (kappa) por cadena. Se promedian
        para predecir y se usan para los intervalos.
    num_chains : int, default=2
        Cadenas MCMC del ajuste final, cada una con su propio burn-in. Con 2
        o más se puede calcular R-hat. Los reajustes intermedios del recorte
        usan una sola cadena: solo aportan residuos.
    num_threads : int, default=1
        Hilos de CPU por cada ajuste de BART. Déjalo en 1 si la validación
        cruzada ya corre en paralelo (n_jobs=-1).
    num_gfr : int, default=0
        Iteraciones de arranque con grow-from-root (XBART). Con 0 es el BART
        clásico del paper. Si es mayor que 0 debe ser >= num_chains: cada
        cadena arranca de un bosque GFR distinto.
    keep_every : int, default=1
        Adelgazamiento: conserva 1 de cada keep_every iteraciones tras el
        burn-in. num_mcmc sigue siendo el número de draws conservados, así
        que el tiempo de cada cadena crece en ese factor.
    random_state : int or None, default=None
        Semilla para que los resultados sean reproducibles.

    Attributes
    ----------
    model_ : stochtree.BARTModel
        Último BART ajustado, entrenado solo con los datos retenidos.
    p_hat_ : float
        Proporción estimada de datos limpios en la última iteración.
    weights_ : ndarray de forma (n,)
        Peso 0/1 de cada fila de entrenamiento en el último ajuste.
    outliers_ : ndarray
        Índices de las filas de entrenamiento recortadas.
    history_ : list of dict
        p_hat, sigma y umbral de cada iteración.
    n_iter_ : int
        Número de iteraciones de recorte realizadas.
    """

    def __init__(
        self,
        Q: float = 1.28,  # noqa: N803 - the paper's name, public API
        trim: bool = True,
        sigma_frac: float = 0.25,
        p_min: float = 0.5,
        tol: float = 1e-4,
        max_iter: int = 20,
        num_trees: int = 50,
        alpha: float = 0.95,
        beta: float = 2.0,
        k: float = 2.0,
        nu: float = 3.0,
        q: float = 0.9,
        num_burnin: int = 500,
        num_mcmc: int = 1000,
        num_chains: int = 2,
        num_threads: int = 1,
        num_gfr: int = 0,
        keep_every: int = 1,
        random_state: Optional[int] = None,
    ) -> None:
        self.Q = Q
        self.trim = trim
        self.sigma_frac = sigma_frac
        self.p_min = p_min
        self.tol = tol
        self.max_iter = max_iter
        self.num_trees = num_trees
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.nu = nu
        self.q = q
        self.num_burnin = num_burnin
        self.num_mcmc = num_mcmc
        self.num_chains = num_chains
        self.num_threads = num_threads
        self.num_gfr = num_gfr
        self.keep_every = keep_every
        self.random_state = random_state

    # --- Priors de Chipman et al. (2010) --------------------------------
    def _chipman_priors(
        self, X: NDArray[np.float64], y: NDArray[np.float64]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Traduce (k, nu, q) a los parámetros de stochtree.

        stochtree estandariza y internamente (media 0, desviación 1), así que
        los priors se calibran en esa escala.

        Returns
        -------
        sigma_prior : dict
            Forma y escala de la IG del prior de sigma^2.
        leaf_prior : dict
            Varianza fija del prior de los valores de las hojas.
        """
        y_std: NDArray[np.float64] = (y - y.mean()) / y.std()

        # sigma^2 ~ nu*lambda / chi2_nu = IG(nu/2, nu*lambda/2), con lambda
        # tal que P(sigma < sigma_hat) = q y sigma_hat = desviación de los
        # residuos de OLS.
        n, p = X.shape
        s2: float
        if n > p + 1:
            A: NDArray[np.float64] = np.column_stack(  # noqa: N806
                [np.ones(n), X]
            )
            beta_ols, *_ = np.linalg.lstsq(A, y_std, rcond=None)
            s2 = float(np.sum((y_std - A @ beta_ols) ** 2) / (n - p - 1))
        else:
            s2 = 1.0
        lam: float = s2 * chi2.ppf(1 - self.q, self.nu) / self.nu

        # mu_hoja ~ N(0, sigma_mu^2), con sigma_mu = rango(y) / (2 k sqrt(m)).
        sigma_mu: float = (y_std.max() - y_std.min()) / (
            2 * self.k * np.sqrt(self.num_trees)
        )

        return (
            {
                "sigma2_global_shape": self.nu / 2,
                "sigma2_global_scale": self.nu * lam / 2,
            },
            {"sigma2_leaf_init": sigma_mu**2, "sample_sigma2_leaf": False},
        )  # fijo, como en Chipman

    # --- BART individual (cadenas MCMC completas) -------------------------
    def _fit_bart(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        num_chains: int = 1,
    ) -> BARTModel:
        """Ajusta un BART clásico con los datos recibidos."""
        seed: int = -1 if self.random_state is None else int(self.random_state)
        sigma_prior, leaf_prior = self._chipman_priors(X, y)
        model: BARTModel = BARTModel()
        model.sample(
            X_train=X,
            y_train=y,
            num_gfr=self.num_gfr,
            num_burnin=self.num_burnin,
            num_mcmc=self.num_mcmc,
            general_params={
                "random_seed": seed,
                "num_chains": num_chains,
                "keep_every": self.keep_every,
                "num_threads": self.num_threads,
                **sigma_prior,
            },
            mean_forest_params={
                "num_trees": self.num_trees,
                "alpha": self.alpha,
                "beta": self.beta,
                **leaf_prior,
            },
        )
        return model

    @staticmethod
    def _draws(
        model: BARTModel, X: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Matriz (n_filas, cadenas * num_mcmc) con f*(x) de cada draw."""
        d: Any = model.predict(X, terms="y_hat")
        return np.asarray(d["y_hat"] if isinstance(d, dict) else d)

    # --- Serialización ----------------------------------------------------
    def __getstate__(self) -> dict[str, Any]:
        """Guarda ``model_`` como el JSON de stochtree.

        Sus bosques viven en objetos de C++ que pickle no puede guardar, así
        que sin esto fallan ``joblib.dump`` y la validación cruzada que
        guarda los folds en disco.
        """
        # Copia: fuera de sklearn, BaseEstimator devuelve el __dict__ vivo.
        state: dict[str, Any] = dict(super().__getstate__())
        if "model_" in state:
            state["model_"] = state["model_"].to_json()
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Reconstruye ``model_`` desde su JSON."""
        if isinstance(state.get("model_"), str):
            model: BARTModel = BARTModel()
            model.from_json(state["model_"])
            state["model_"] = model
        super().__setstate__(state)

    # --- API de scikit-learn ----------------------------------------------
    def fit(self, X: ArrayLike, y: ArrayLike) -> BARTAtrRegressor:
        """Ajusta BART(Atr) (o BART(Def) si trim=False).

        Parameters
        ----------
        X : array-like de forma (n, p)
            Covariables numéricas (ya preprocesadas).
        y : array-like de forma (n,)
            Respuesta.

        Returns
        -------
        self : BARTAtrRegressor
            El estimador ajustado.
        """
        # stochtree lo exige en el ajuste final; mejor fallar antes del
        # recorte que tras varios minutos de reajustes.
        if 0 < self.num_gfr < self.num_chains:
            raise ValueError(
                "num_gfr debe ser 0 o >= num_chains: cada cadena arranca de "
                "un draw GFR distinto."
            )
        X_arr: NDArray[np.float64]
        y_arr: NDArray[np.float64]
        X_arr, y_arr = validate_data(
            self, X, y, y_numeric=True, dtype=np.float64
        )
        n: int = len(y_arr)
        z: NDArray[np.float64] = folded_normal_quantiles(n)

        # Solo el ajuste final lleva num_chains cadenas. Con recorte, el
        # inicial nunca es el final: el bucle reajusta al menos una vez.
        initial_is_final: bool = not self.trim or self.max_iter < 1

        # Paso 1: BART inicial con todos los datos
        model: BARTModel = self._fit_bart(
            X_arr, y_arr, self.num_chains if initial_is_final else 1
        )
        weights: NDArray[np.int64] = np.ones(n, dtype=int)
        history: list[dict[str, float]] = []
        p_old: float = 0.5
        p_hat: float = 1.0

        if self.trim:
            for it in range(1, self.max_iter + 1):
                resid: NDArray[np.float64] = y_arr - self._draws(
                    model, X_arr
                ).mean(axis=1)
                abs_sorted: NDArray[np.float64] = np.sort(np.abs(resid))
                sigma2: float = robust_sigma2(
                    abs_sorted, z, self.sigma_frac
                )  # paso 2
                p_hat, _ = estimate_p(abs_sorted, z, sigma2, self.Q)  # paso 3
                # Salvaguarda (no está en el paper): nunca recortar más de
                # 1 - p_min.
                p_hat = max(p_hat, self.p_min)
                k_pos: int = max(
                    1, int(np.floor(n * p_hat))
                )  # posición crítica
                threshold: float = float(
                    abs_sorted[k_pos - 1]
                )  # su valor = umbral
                weights = (np.abs(resid) <= threshold).astype(
                    int
                )  # paso 4, ec. (8)

                # El criterio de parada (paso 5).
                final: bool = (
                    abs(p_hat - p_old) / p_hat < self.tol
                    or it == self.max_iter
                )
                model = self._fit_bart(
                    X_arr[weights == 1],
                    y_arr[weights == 1],
                    self.num_chains if final else 1,
                )
                history.append(
                    {
                        "iter": it,
                        "p_hat": p_hat,
                        "sigma": float(np.sqrt(sigma2)),
                        "threshold": threshold,
                    }
                )
                if final:  # paso 5
                    break
                p_old = p_hat

        self.model_ = model
        self.p_hat_ = p_hat
        self.weights_ = weights
        self.outliers_ = np.flatnonzero(weights == 0)
        self.history_ = history
        self.n_iter_ = len(history)
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        """Media posterior de f(x) con el último BART.

        Parameters
        ----------
        X : array-like de forma (n, p)

        Returns
        -------
        y_pred : ndarray de forma (n,)
        """
        check_is_fitted(self, "model_")
        X_arr: NDArray[np.float64] = validate_data(
            self, X, reset=False, dtype=np.float64
        )
        return self._draws(self.model_, X_arr).mean(axis=1)

    def posterior_draws(self, X: ArrayLike) -> NDArray[np.float64]:
        """Draws posteriores de f(x) separados por cadena.

        Parameters
        ----------
        X : array-like de forma (n, p)

        Returns
        -------
        draws : ndarray de forma (cadenas, num_mcmc, n)
            El orden (chain, draw, ...) que espera arviz.
        """
        check_is_fitted(self, "model_")
        X_arr: NDArray[np.float64] = validate_data(
            self, X, reset=False, dtype=np.float64
        )
        draws: NDArray[np.float64] = self._draws(self.model_, X_arr)
        n_chains: int = self.model_.num_chains
        return draws.T.reshape(n_chains, -1, len(X_arr))

    def predict_interval(
        self,
        X: ArrayLike,
        level: float = 0.95,
        kind: Literal["mean", "prediction"] = "prediction",
    ) -> NDArray[np.float64]:
        """Intervalos de credibilidad.

        Parameters
        ----------
        X : array-like de forma (n, p)
        level : float, default=0.95
            Nivel de credibilidad.
        kind : {"mean", "prediction"}, default="prediction"
            "mean" da el intervalo para f(x); "prediction" el intervalo para
            una y nueva del proceso limpio (suma ruido N(0, sigma^2)).

        Returns
        -------
        bounds : ndarray de forma (2, n)
            Fila 0: límite inferior. Fila 1: límite superior.
        """
        check_is_fitted(self, "model_")
        X_arr: NDArray[np.float64] = validate_data(
            self, X, reset=False, dtype=np.float64
        )
        draws: NDArray[np.float64] = self._draws(self.model_, X_arr)
        if kind == "prediction":
            sigma: NDArray[np.float64] = np.sqrt(
                self.model_.global_var_samples
            )
            rng: np.random.Generator = np.random.default_rng(self.random_state)
            draws = draws + rng.normal(size=draws.shape) * sigma[None, :]
        a: float = (1 - level) / 2 * 100
        return np.percentile(draws, [a, 100 - a], axis=1)
