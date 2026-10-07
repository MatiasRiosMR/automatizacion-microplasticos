"""Clasificador de partículas contra los clusters de referencia de los 6 polímeros.

Asigna cada partícula (representada por sus coordenadas de phasor) a uno de los 6
polímeros, o a la categoría ``"no_clasificable"`` cuando cae fuera de todos los clusters
conocidos.

Estrategias
-----------
- ``"centroide"``: cluster de mayor verosimilitud gaussiana (análisis discriminante
  cuadrático, QDA): ``argmin(d² + log|Σ|)``, con ``d²`` la distancia de Mahalanobis a cada
  cluster. El término ``log|Σ|`` evita que un cluster disperso (p. ej. un polímero más
  envejecido) «robe» partículas de clusters compactos vecinos. ``regla="mahalanobis"``
  recupera la asignación por distancia pura (comportamiento anterior a la auditoría).
  Es la línea base, análoga al vecino más cercano multivariado de FIMAP (Ho et al. 2025).
- ``"knn"``: voto de los ``k`` vecinos más cercanos entre las mediciones de calibración,
  ponderado por ``1/distancia`` (el desempate favorece al vecino más cercano).
- ``"gmm"``: una gaussiana por polímero con peso proporcional a su número de mediciones
  (mezcla de gaussianas **supervisada**). Etiqueta y score salen del mismo componente.

Regla de "no clasificable" (crítica para falsos positivos)
--------------------------------------------------------
El parámetro ``confianza`` (por defecto ``0.99``) fija la fracción de partículas de un
polímero de referencia que se espera **aceptar**.

- ``centroide`` / ``gmm``: una partícula se rechaza si su ``d²`` al cluster asignado supera
  el umbral de ese cluster. Con ``umbral="hotelling"`` (por defecto) el umbral es la
  **región de predicción de Hotelling**, que tiene en cuenta que media y covarianza se
  estimaron con ``n`` mediciones::

      d² ≤ (n + 1)/n · p(n − 1)/(n − p) · F₁₋α(p, n − p)

  Con ``umbral="chi2"`` se usa ``chi2.ppf(confianza, p)``, que supone media y covarianza
  exactas: con pocas mediciones rechaza muchas más partículas reales de lo prometido
  (auditoría 2026-10: 23 % de falso rechazo en fusión 4D con ``n = 10`` contra 1 % nominal;
  ver ``docs/AUDITORIA.md``). Ambos convergen cuando ``n`` es grande.
- ``knn``: con ``umbral_knn="conformal"`` el score es la distancia media a los ``k``
  vecinos **de la clase asignada** y el umbral es el cuantil conformal (leave-one-out) de
  ese estadístico en la calibración, de modo que ``confianza`` es una cobertura real. Con
  ``umbral_knn="cuantil"`` (por defecto, compatibilidad) el umbral es el cuantil
  ``confianza`` de la distancia intra-clase multiplicado por ``margen_knn``.

``confianza=None`` desactiva el rechazo (todo se asigna al polímero más cercano).

Las partículas con phasor no finito (``NaN``: ROI sin fotones suficientes) se informan
siempre como ``"no_clasificable"`` con score ``inf``, en las tres estrategias.

Esta regla es lo que separa la señal de Nile Red-MP de la materia orgánica fluorescente
(muestras ambientales) y de la autofluorescencia celular (monocitos/neutrófilos): esas
señales no forman parte de ningún cluster de polímero y deberían caer fuera del umbral.
El umbral de distancia no alcanza para separar autofluorescencia que cae muy cerca de un
polímero; ese caso requiere evidencia adicional (ver ``docs/AUDITORIA.md``).
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import chi2
from scipy.stats import f as dist_f
from sklearn.neighbors import NearestNeighbors

from . import NO_CLASIFICABLE
from .calibracion import Calibracion

ESTRATEGIAS = ("centroide", "knn", "gmm")
REGLAS = ("qda", "mahalanobis")
UMBRALES = ("hotelling", "chi2")
UMBRALES_KNN = ("cuantil", "conformal")


def _mahalanobis_cuadrado(x: np.ndarray, media: np.ndarray, cov_inv: np.ndarray) -> np.ndarray:
    """Distancia de Mahalanobis al cuadrado de cada fila de ``x`` respecto de ``media``.

    Parameters
    ----------
    x : numpy.ndarray, shape (n, d)
    media : numpy.ndarray, shape (d,)
    cov_inv : numpy.ndarray, shape (d, d)
        Inversa de la matriz de covarianza del cluster.

    Returns
    -------
    numpy.ndarray, shape (n,)
    """
    delta = x - media
    return np.einsum("ni,ij,nj->n", delta, cov_inv, delta)


def _inversa_regularizada(cov: np.ndarray, d: int, epsilon: float) -> tuple[np.ndarray, float]:
    """Inversa y log-determinante de una covarianza regularizada en la diagonal."""
    cov = np.atleast_2d(np.asarray(cov, dtype=float)) + np.eye(d) * epsilon
    return np.linalg.inv(cov), float(np.linalg.slogdet(cov)[1])


def umbral_hotelling(n: int, p: int, confianza: float) -> float:
    """Umbral de ``d²`` de la región de predicción de Hotelling para una observación nueva.

    Parameters
    ----------
    n : int
        Mediciones con que se estimaron media y covarianza del cluster.
    p : int
        Dimensión del espacio de phasores.
    confianza : float
        Fracción de observaciones nuevas del cluster que debe quedar dentro.

    Returns
    -------
    float
        ``(n+1)/n · p(n−1)/(n−p) · F⁻¹(confianza; p, n−p)``. ``inf`` si ``n <= p`` (no hay
        información suficiente para acotar la región).

    Notes
    -----
    Si ``x̄`` y ``S`` son la media y la covarianza muestrales de ``n`` observaciones
    gaussianas ``p``-dimensionales, una observación nueva ``x`` cumple
    ``n/(n+1) · (x − x̄)ᵀ S⁻¹ (x − x̄) ~ p(n−1)/(n−p) · F(p, n−p)`` (Hotelling, 1931).
    """
    if n <= p:
        return float("inf")
    return (n + 1) / n * p * (n - 1) / (n - p) * float(dist_f.ppf(confianza, p, n - p))


class ClasificadorPhasor:
    """Clasifica partículas en el espacio de phasores contra una :class:`Calibracion`.

    Parameters
    ----------
    calibracion : Calibracion
        Firma de referencia de los polímeros.
    estrategia : {"centroide", "knn", "gmm"}, optional
        Método de clasificación. Por defecto ``"centroide"``.
    confianza : float or None, optional
        Fracción de partículas de un polímero de referencia que se espera aceptar, en
        ``(0, 1)``. Por defecto ``0.99``. ``None`` desactiva el rechazo.
    k : int, optional
        Número de vecinos para ``estrategia="knn"``. Por defecto ``5``.
    margen_knn : float, optional
        Factor sobre el cuantil de calibración para ``umbral_knn="cuantil"``. Por defecto ``1.5``.
    regularizacion_cov : float, optional
        Valor sumado a la diagonal de cada covarianza antes de invertirla. Por defecto ``1e-6``.
    regla : {"qda", "mahalanobis"}, optional
        Regla de asignación de ``centroide``. Por defecto ``"qda"`` (``d² + log|Σ|``).
    umbral : {"hotelling", "chi2"}, optional
        Umbral de rechazo de ``centroide``/``gmm``. Por defecto ``"hotelling"``.
    umbral_knn : {"cuantil", "conformal"}, optional
        Umbral de rechazo de ``knn``. Por defecto ``"cuantil"``.

    Attributes
    ----------
    etiquetas_ : list[str]
        Polímeros conocidos, en orden.
    umbral_mahalanobis2_ : float
        Umbral χ² de referencia (``chi2.ppf(confianza, df)``), independiente de ``n``.
    umbrales_ : dict[str, float]
        Umbral de ``d²`` efectivo de cada polímero (Hotelling o χ² según ``umbral``).
    umbral_knn_ : float or None
        Umbral global de ``knn`` con ``umbral_knn="cuantil"`` (tras ``entrenar``).
    """

    def __init__(
        self,
        calibracion: Calibracion,
        estrategia: str = "centroide",
        confianza: float | None = 0.99,
        k: int = 5,
        margen_knn: float = 1.5,
        regularizacion_cov: float = 1e-6,
        regla: str = "qda",
        umbral: str = "hotelling",
        umbral_knn: str = "cuantil",
    ) -> None:
        if estrategia not in ESTRATEGIAS:
            raise ValueError(f"estrategia debe ser una de {ESTRATEGIAS}, no {estrategia!r}")
        if confianza is not None and not (0.0 < confianza < 1.0):
            raise ValueError(f"confianza debe estar en (0, 1) o ser None, no {confianza!r}")
        if regla not in REGLAS:
            raise ValueError(f"regla debe ser una de {REGLAS}, no {regla!r}")
        if umbral not in UMBRALES:
            raise ValueError(f"umbral debe ser uno de {UMBRALES}, no {umbral!r}")
        if umbral_knn not in UMBRALES_KNN:
            raise ValueError(f"umbral_knn debe ser uno de {UMBRALES_KNN}, no {umbral_knn!r}")

        self.calibracion = calibracion
        self.estrategia = estrategia
        self.confianza = confianza
        self.k = k
        self.margen_knn = margen_knn
        self.regularizacion_cov = regularizacion_cov
        self.regla = regla
        self.umbral = umbral
        self.umbral_knn = umbral_knn

        self.etiquetas_: list[str] = calibracion.etiquetas
        d = calibracion.n_features
        self.umbral_mahalanobis2_ = (
            float(chi2.ppf(confianza, df=d)) if confianza is not None else np.inf
        )
        self._ajustar_gaussianas(
            [calibracion.centroides[e] for e in self.etiquetas_],
            [calibracion.covarianzas[e] for e in self.etiquetas_],
            [calibracion.n_muestras.get(e, 0) for e in self.etiquetas_],
        )
        self.umbral_knn_: float | None = None
        self._umbral_knn_clase: dict[str, float] = {}
        self._nn: NearestNeighbors | None = None
        self._nn_X: np.ndarray | None = None
        self._nn_etiquetas: np.ndarray | None = None
        self._nn_por_clase: dict[str, tuple[NearestNeighbors, int]] = {}
        self._entrenado = estrategia != "knn"

    # ------------------------------------------------------------------ gaussianas
    def _ajustar_gaussianas(self, medias, covs, ns) -> None:
        d = self.calibracion.n_features
        self._medias = np.vstack([np.asarray(m, dtype=float) for m in medias])
        inv_logdet = [_inversa_regularizada(c, d, self.regularizacion_cov) for c in covs]
        self._cov_inv = [il[0] for il in inv_logdet]
        self._logdet = np.array([il[1] for il in inv_logdet])
        ns = np.asarray(ns, dtype=float)
        self._log_peso = (np.log(ns / ns.sum()) if (ns > 0).all() else
                          np.full(len(ns), -np.log(len(ns))))
        self.umbrales_: dict[str, float] = {}
        for etiqueta, n in zip(self.etiquetas_, ns):
            if self.confianza is None:
                u = np.inf
            elif self.umbral == "hotelling" and n > 0:
                u = umbral_hotelling(int(n), d, self.confianza)
                if not np.isfinite(u):
                    warnings.warn(
                        f"'{etiqueta}' tiene {int(n)} mediciones para {d} dimensiones: la región "
                        "de aceptación no está acotada y no se rechazará ninguna partícula "
                        "asignada a ese polímero. Se necesitan al menos d + 1 mediciones.",
                        stacklevel=3,
                    )
            else:
                u = self.umbral_mahalanobis2_
            self.umbrales_[etiqueta] = float(u)
        self._umbrales = np.array([self.umbrales_[e] for e in self.etiquetas_])

    # ------------------------------------------------------------------ entrenamiento
    def entrenar(self, X: np.ndarray | None = None, y: np.ndarray | None = None) -> ClasificadorPhasor:
        """Ajusta las estructuras internas según la estrategia.

        Parameters
        ----------
        X : numpy.ndarray, shape (n, d), optional
            Mediciones individuales de calibración. **Obligatorio** para ``knn``; para
            ``gmm`` (junto con ``y``) re-estima cada gaussiana y su peso; ignorado por
            ``centroide``.
        y : numpy.ndarray, shape (n,), optional
            Etiqueta de polímero de cada fila de ``X``.

        Returns
        -------
        ClasificadorPhasor
            ``self``, para encadenar.
        """
        if self.estrategia == "knn":
            return self._entrenar_knn(X, y)
        if self.estrategia == "gmm" and X is not None:
            self._entrenar_gmm(X, y)
        self._entrenado = True
        return self

    def _entrenar_gmm(self, X, y) -> None:
        if y is None:
            raise ValueError(
                "La estrategia 'gmm' es supervisada: pasá X e y (etiqueta de cada medición). "
                "Sin argumentos usa los centroides y covarianzas de la calibración."
            )
        X, y = _filas_finitas(np.asarray(X, dtype=float), np.asarray(y).astype(str))
        medias, covs, ns = [], [], []
        for etiqueta in self.etiquetas_:
            Xc = X[y == etiqueta]
            if len(Xc) < 2:
                raise ValueError(f"'{etiqueta}' necesita al menos 2 mediciones para 'gmm'.")
            medias.append(Xc.mean(axis=0))
            covs.append(np.cov(Xc, rowvar=False))
            ns.append(len(Xc))
        self._ajustar_gaussianas(medias, covs, ns)

    def _entrenar_knn(self, X, y) -> ClasificadorPhasor:
        if X is None or y is None:
            raise ValueError("La estrategia 'knn' necesita X e y con las mediciones de calibración.")
        X, y = _filas_finitas(np.asarray(X, dtype=float), np.asarray(y).astype(str))
        self._nn = NearestNeighbors(n_neighbors=min(self.k, len(X))).fit(X)
        self._nn_X, self._nn_etiquetas = X, y

        distancias_intra = []
        for etiqueta in np.unique(y):
            Xc = X[y == etiqueta]
            if len(Xc) <= 1:
                continue
            kk = min(self.k, len(Xc) - 1)
            nn_c = NearestNeighbors(n_neighbors=kk + 1).fit(Xc)
            self._nn_por_clase[str(etiqueta)] = (nn_c, kk)
            dist, _ = nn_c.kneighbors(Xc)
            loo = dist[:, 1:].mean(axis=1)  # leave-one-out: excluye el propio punto
            distancias_intra.append(loo)
            if self.confianza is not None:
                nivel = min(1.0, np.ceil((len(loo) + 1) * self.confianza) / len(loo))
                self._umbral_knn_clase[str(etiqueta)] = float(np.quantile(loo, nivel))
        if self.confianza is not None and distancias_intra:
            todas = np.concatenate(distancias_intra)
            self.umbral_knn_ = float(np.quantile(todas, self.confianza) * self.margen_knn)
        self._entrenado = True
        return self

    # ------------------------------------------------------------------ predicción
    def predecir(self, X: np.ndarray) -> np.ndarray:
        """Clasifica cada fila de ``X``.

        Parameters
        ----------
        X : numpy.ndarray, shape (n, d)
            Coordenadas de phasor de las partículas a clasificar.

        Returns
        -------
        numpy.ndarray of str, shape (n,)
            Código de polímero, o ``"no_clasificable"``.
        """
        etiquetas, _ = self.predecir_con_score(X)
        return etiquetas

    def predecir_con_score(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Como :meth:`predecir` pero devuelve también el score de rechazo.

        Returns
        -------
        etiquetas : numpy.ndarray of str, shape (n,)
        score : numpy.ndarray, shape (n,)
            Cociente ``estadístico / umbral``: valores ``> 1`` son ``"no_clasificable"``.
            Para ``centroide``/``gmm`` es ``d² / umbral`` del cluster asignado; para
            ``knn`` es la distancia media a los vecinos dividida por el umbral. Las filas
            con phasor no finito tienen score ``inf``.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.calibracion.n_features:
            raise ValueError(
                f"X debe tener forma (n, {self.calibracion.n_features}); recibí {X.shape}."
            )
        if not self._entrenado:
            raise RuntimeError("Llamá a entrenar(X, y) antes de predecir con estrategia 'knn'.")

        finito = np.isfinite(X).all(axis=1)
        etiquetas = np.full(len(X), NO_CLASIFICABLE, dtype=object)
        score = np.full(len(X), np.inf)
        if finito.any():
            Xf = X[finito]
            if self.estrategia == "knn":
                e, s = self._predecir_knn(Xf)
            else:
                e, s = self._predecir_gaussiano(Xf)
            etiquetas[finito] = e
            score[finito] = s
        if self.confianza is not None:
            etiquetas[score > 1.0] = NO_CLASIFICABLE
        return etiquetas.astype(str), score

    def _predecir_gaussiano(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``centroide`` (QDA o Mahalanobis) y ``gmm`` (QDA con pesos por clase)."""
        d2 = np.column_stack([
            _mahalanobis_cuadrado(X, m, ci) for m, ci in zip(self._medias, self._cov_inv)
        ])
        if self.estrategia == "gmm":
            criterio = d2 + self._logdet - 2 * self._log_peso
        elif self.regla == "qda":
            criterio = d2 + self._logdet
        else:
            criterio = d2
        j = criterio.argmin(axis=1)
        etiqueta = np.array(self.etiquetas_, dtype=object)[j]
        score = d2[np.arange(len(X)), j] / self._umbrales[j]
        return etiqueta, score

    def _predecir_knn(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Voto de los k vecinos ponderado por 1/distancia; score según ``umbral_knn``."""
        dist, idx = self._nn.kneighbors(X)
        etiquetas_vecinos = self._nn_etiquetas[idx]
        etiqueta = np.array(
            [_voto_ponderado(e, d) for e, d in zip(etiquetas_vecinos, dist)], dtype=object
        )
        if self.confianza is None:
            return etiqueta, np.zeros(len(X))
        if self.umbral_knn == "cuantil":
            return etiqueta, dist.mean(axis=1) / self.umbral_knn_
        score = np.empty(len(X))
        for i, e in enumerate(etiqueta):
            nn_c, kk = self._nn_por_clase[e]
            dc, _ = nn_c.kneighbors(X[i : i + 1], n_neighbors=kk)
            score[i] = dc.mean() / self._umbral_knn_clase[e]
        return etiqueta, score


def _filas_finitas(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Descarta mediciones de calibración con phasor no finito."""
    ok = np.isfinite(X).all(axis=1)
    return X[ok], y[ok]


def _voto_ponderado(etiquetas: np.ndarray, distancias: np.ndarray):
    """Etiqueta con mayor suma de pesos ``1/distancia``.

    Un vecino a distancia 0 decide solo. A igualdad de peso gana la etiqueta del vecino
    más cercano (``etiquetas`` viene ordenado por distancia creciente).
    """
    pesos = 1.0 / np.maximum(np.asarray(distancias, dtype=float), 1e-12)
    votos: dict[str, float] = {}
    for etiqueta, peso in zip(etiquetas, pesos):
        votos[etiqueta] = votos.get(etiqueta, 0.0) + peso
    maximo = max(votos.values())
    return next(e for e in etiquetas if np.isclose(votos[e], maximo))
