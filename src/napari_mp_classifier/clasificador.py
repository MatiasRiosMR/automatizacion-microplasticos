"""Clasificador de partículas contra los clusters de referencia de los 6 polímeros.

Asigna cada partícula (representada por sus coordenadas de phasor) a uno de los 6
polímeros, o a la categoría ``"no_clasificable"`` cuando cae fuera de todos los clusters
conocidos.

Estrategias
-----------
- ``"centroide"``: centroide más cercano por distancia de Mahalanobis (usa la covarianza
  de cada cluster de la calibración). Es la línea base, equivalente al vecino más cercano
  multivariado de FIMAP (Ho et al. 2025).
- ``"knn"``: k vecinos más cercanos sobre las mediciones de calibración.
- ``"gmm"``: mezcla de gaussianas, un componente por polímero, inicializada con la
  calibración; el score usa las covarianzas ajustadas.

Regla de "no clasificable" (crítica para falsos positivos)
--------------------------------------------------------
El parámetro ``confianza`` (por defecto ``0.99``) fija un umbral **estadístico y
consciente de la dimensión**:

- ``centroide`` / ``gmm``: una partícula se rechaza si su distancia de Mahalanobis al
  cuadrado al cluster más cercano supera ``chi2.ppf(confianza, df=n_features)``. Es decir,
  se acepta solo lo que cae dentro de la región que concentra el ``confianza`` de la masa
  de probabilidad de un cluster gaussiano. Esto corrige el problema de usar un umbral fijo
  en σ: en 4D (fusión FLIM+espectral) la distancia de Mahalanobis crece respecto a 2D, y
  un umbral fijo rechazaría de más.
- ``knn``: se estima la distribución de distancias intra-clase de la calibración y se
  rechaza lo que supera su cuantil ``confianza`` (con un pequeño margen).

``confianza=None`` desactiva el rechazo (todo se asigna al polímero más cercano).

Esta regla es lo que separa la señal de Nile Red-MP de la materia orgánica fluorescente
(muestras ambientales) y de la autofluorescencia celular (monocitos/neutrófilos): esas
señales no forman parte de ningún cluster de polímero y deben caer fuera del umbral.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

from . import NO_CLASIFICABLE
from .calibracion import Calibracion

ESTRATEGIAS = ("centroide", "knn", "gmm")


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


def _inversa_regularizada(cov: np.ndarray, d: int, epsilon: float) -> np.ndarray:
    """Inversa de una covarianza con regularización en la diagonal (evita singularidad)."""
    cov = np.atleast_2d(np.asarray(cov, dtype=float)) + np.eye(d) * epsilon
    return np.linalg.inv(cov)


class ClasificadorPhasor:
    """Clasifica partículas en el espacio de phasores contra una :class:`Calibracion`.

    Parameters
    ----------
    calibracion : Calibracion
        Firma de referencia de los polímeros.
    estrategia : {"centroide", "knn", "gmm"}, optional
        Método de clasificación. Por defecto ``"centroide"``.
    confianza : float or None, optional
        Nivel de confianza para la regla de "no clasificable", en ``(0, 1)``.
        Por defecto ``0.99``. ``None`` desactiva el rechazo.
    k : int, optional
        Número de vecinos para ``estrategia="knn"``. Por defecto ``5``.
    margen_knn : float, optional
        Factor multiplicativo sobre el cuantil de calibración para el umbral de ``knn``.
        Por defecto ``1.5``.
    regularizacion_cov : float, optional
        Valor sumado a la diagonal de cada covarianza antes de invertirla. Por defecto ``1e-6``.

    Attributes
    ----------
    etiquetas_ : list[str]
        Polímeros conocidos, en orden.
    umbral_mahalanobis2_ : float
        Umbral de distancia de Mahalanobis al cuadrado (``chi2.ppf(confianza, df)``).
    umbral_knn_ : float or None
        Umbral de distancia media a los k vecinos (solo tras ``entrenar`` con ``knn``).
    """

    def __init__(
        self,
        calibracion: Calibracion,
        estrategia: str = "centroide",
        confianza: float | None = 0.99,
        k: int = 5,
        margen_knn: float = 1.5,
        regularizacion_cov: float = 1e-6,
    ) -> None:
        if estrategia not in ESTRATEGIAS:
            raise ValueError(f"estrategia debe ser una de {ESTRATEGIAS}, no {estrategia!r}")
        if confianza is not None and not (0.0 < confianza < 1.0):
            raise ValueError(f"confianza debe estar en (0, 1) o ser None, no {confianza!r}")

        self.calibracion = calibracion
        self.estrategia = estrategia
        self.confianza = confianza
        self.k = k
        self.margen_knn = margen_knn
        self.regularizacion_cov = regularizacion_cov

        self.etiquetas_: list[str] = calibracion.etiquetas
        self._medias = calibracion.matriz_centroides()  # (n_polimeros, d)
        d = calibracion.n_features

        self._cov_inv: dict[str, np.ndarray] = {
            etiqueta: _inversa_regularizada(
                calibracion.covarianzas[etiqueta], d, regularizacion_cov
            )
            for etiqueta in self.etiquetas_
        }

        self.umbral_mahalanobis2_ = (
            float(chi2.ppf(confianza, df=d)) if confianza is not None else np.inf
        )
        self.umbral_knn_: float | None = None

        self._nn: NearestNeighbors | None = None
        self._nn_etiquetas: np.ndarray | None = None
        self._gmm: GaussianMixture | None = None
        self._gmm_a_etiqueta: np.ndarray | None = None
        self._gmm_cov_inv: list[np.ndarray] | None = None

    # ------------------------------------------------------------------ entrenamiento
    def entrenar(self, X: np.ndarray | None = None, y: np.ndarray | None = None) -> "ClasificadorPhasor":
        """Ajusta las estructuras internas según la estrategia.

        Parameters
        ----------
        X : numpy.ndarray, shape (n, d), optional
            Mediciones individuales de calibración. **Obligatorio** para ``knn``; opcional
            para ``gmm`` (mejora la estimación); ignorado por ``centroide``.
        y : numpy.ndarray, shape (n,), optional
            Etiqueta de polímero de cada fila de ``X``. Obligatorio junto con ``X``.

        Returns
        -------
        ClasificadorPhasor
            ``self``, para encadenar.
        """
        if self.estrategia == "centroide":
            return self

        if self.estrategia == "knn":
            return self._entrenar_knn(X, y)

        return self._entrenar_gmm(X)

    def _entrenar_knn(self, X, y) -> "ClasificadorPhasor":
        if X is None or y is None:
            raise ValueError("La estrategia 'knn' necesita X e y con las mediciones de calibración.")
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self._nn = NearestNeighbors(n_neighbors=min(self.k, len(X)))
        self._nn.fit(X)
        self._nn_etiquetas = y

        # Umbral: cuantil `confianza` de la distancia media intra-clase en la calibración.
        if self.confianza is not None:
            distancias_intra = []
            for etiqueta in np.unique(y):
                Xc = X[y == etiqueta]
                if len(Xc) <= 1:
                    continue
                kk = min(self.k, len(Xc) - 1)
                nn_c = NearestNeighbors(n_neighbors=kk + 1).fit(Xc)
                dist, _ = nn_c.kneighbors(Xc)
                distancias_intra.append(dist[:, 1:].mean(axis=1))  # excluye el propio punto
            todas = np.concatenate(distancias_intra)
            self.umbral_knn_ = float(np.quantile(todas, self.confianza) * self.margen_knn)
        return self

    def _entrenar_gmm(self, X) -> "ClasificadorPhasor":
        n = len(self.etiquetas_)
        self._gmm = GaussianMixture(
            n_components=n,
            covariance_type="full",
            means_init=self._medias,
            reg_covar=max(self.regularizacion_cov, 1e-6),
            random_state=0,
        )
        if X is not None:
            self._gmm.fit(np.asarray(X, dtype=float))
        else:
            rng = np.random.default_rng(0)
            muestras = np.vstack(
                [
                    rng.multivariate_normal(
                        self.calibracion.centroides[e], self.calibracion.covarianzas[e], size=300
                    )
                    for e in self.etiquetas_
                ]
            )
            self._gmm.fit(muestras)

        # Mapear cada componente ajustado al polímero de centroide más cercano.
        d = self.calibracion.n_features
        self._gmm_a_etiqueta = np.array(
            [
                self.etiquetas_[int(np.argmin(np.linalg.norm(self._medias - m, axis=1)))]
                for m in self._gmm.means_
            ],
            dtype=object,
        )
        self._gmm_cov_inv = [
            _inversa_regularizada(cov, d, self.regularizacion_cov)
            for cov in self._gmm.covariances_
        ]
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
            Cociente ``score / umbral``: valores ``> 1`` son ``"no_clasificable"``.
            Para ``centroide``/``gmm`` es ``mahalanobis2 / chi2.ppf(confianza, df)``;
            para ``knn`` es ``dist_media_k / umbral_knn``. Independiente de la dimensión,
            así se puede comparar entre estrategias y modalidades.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.calibracion.n_features:
            raise ValueError(
                f"X debe tener forma (n, {self.calibracion.n_features}); recibí {X.shape}."
            )

        if self.estrategia == "knn":
            etiqueta_cruda, score = self._predecir_knn(X)
        elif self.estrategia == "gmm":
            etiqueta_cruda, score = self._predecir_gmm(X)
        else:
            etiqueta_cruda, score = self._predecir_centroide(X)

        etiquetas = etiqueta_cruda.astype(object)
        if self.confianza is not None:
            etiquetas[score > 1.0] = NO_CLASIFICABLE
        return etiquetas.astype(str), score

    def _predecir_centroide(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Centroide más cercano por distancia de Mahalanobis (score normalizado por chi²)."""
        n = len(X)
        d2 = np.empty((n, len(self.etiquetas_)))
        for j, etiqueta in enumerate(self.etiquetas_):
            d2[:, j] = _mahalanobis_cuadrado(X, self._medias[j], self._cov_inv[etiqueta])
        j_min = d2.argmin(axis=1)
        etiqueta = np.array(self.etiquetas_, dtype=object)[j_min]
        score = d2[np.arange(n), j_min] / self.umbral_mahalanobis2_
        return etiqueta, score

    def _predecir_gmm(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Componente GMM de mayor responsabilidad; score por Mahalanobis a ese componente."""
        if self._gmm is None or self._gmm_a_etiqueta is None or self._gmm_cov_inv is None:
            raise RuntimeError("Llamá a entrenar() antes de predecir con estrategia 'gmm'.")
        componente = self._gmm.predict(X)
        etiqueta = self._gmm_a_etiqueta[componente]

        n = len(X)
        d2 = np.empty((n, self._gmm.n_components))
        for j in range(self._gmm.n_components):
            d2[:, j] = _mahalanobis_cuadrado(X, self._gmm.means_[j], self._gmm_cov_inv[j])
        score = d2.min(axis=1) / self.umbral_mahalanobis2_
        return etiqueta, score

    def _predecir_knn(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Voto mayoritario de los k vecinos; score = dist media / umbral_knn."""
        if self._nn is None or self._nn_etiquetas is None:
            raise RuntimeError("Llamá a entrenar(X, y) antes de predecir con estrategia 'knn'.")
        dist, idx = self._nn.kneighbors(X)
        etiquetas_vecinos = self._nn_etiquetas[idx]
        etiqueta = np.array(
            [_voto_mayoritario(fila) for fila in etiquetas_vecinos], dtype=object
        )
        dist_media = dist.mean(axis=1)
        if self.umbral_knn_ is not None:
            score = dist_media / self.umbral_knn_
        else:
            score = np.zeros(len(X))
        return etiqueta, score


def _voto_mayoritario(etiquetas: np.ndarray):
    """Etiqueta más frecuente en ``etiquetas`` (desempate por primera aparición)."""
    valores, cuentas = np.unique(etiquetas, return_counts=True)
    return valores[cuentas.argmax()]
