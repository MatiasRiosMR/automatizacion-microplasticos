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
  calibración.

Regla de "no clasificable" (crítica para falsos positivos)
--------------------------------------------------------
Una partícula queda ``"no_clasificable"`` cuando su score de pertenencia al mejor cluster
supera ``umbral_no_clasificable``:

- ``centroide`` / ``gmm``: distancia de Mahalanobis al centroide más cercano, en unidades
  de desvío estándar. Con ``umbral=3.0`` se rechaza lo que está a más de ~3 σ de todo
  cluster (regla habitual: fuera del 99% de la masa del cluster gaussiano).
- ``knn``: distancia euclídea media a los k vecinos, comparada con el umbral.

Esto es lo que separa la señal de Nile Red-MP de la materia orgánica fluorescente
(muestras ambientales) y de la autofluorescencia celular (monocitos/neutrófilos):
esas señales no forman parte de ningún cluster de polímero y deben caer fuera del umbral.
"""

from __future__ import annotations

import numpy as np
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


class ClasificadorPhasor:
    """Clasifica partículas en el espacio de phasores contra una :class:`Calibracion`.

    Parameters
    ----------
    calibracion : Calibracion
        Firma de referencia de los polímeros.
    estrategia : {"centroide", "knn", "gmm"}, optional
        Método de clasificación. Por defecto ``"centroide"``.
    umbral_no_clasificable : float, optional
        Umbral de rechazo. Para ``centroide``/``gmm`` es la distancia de Mahalanobis
        máxima (en σ) al centroide más cercano; por defecto ``3.0``. Para ``knn`` es la
        distancia euclídea media máxima a los ``k`` vecinos; conviene fijarla a partir de
        la dispersión de la calibración. ``None`` desactiva el rechazo (todo se asigna al
        polímero más cercano).
    k : int, optional
        Número de vecinos para ``estrategia="knn"``. Por defecto ``5``.
    regularizacion_cov : float, optional
        Valor sumado a la diagonal de cada covarianza antes de invertirla, para evitar
        matrices singulares con pocos datos. Por defecto ``1e-6``.

    Attributes
    ----------
    etiquetas_ : list[str]
        Polímeros conocidos, en orden.
    """

    def __init__(
        self,
        calibracion: Calibracion,
        estrategia: str = "centroide",
        umbral_no_clasificable: float | None = 3.0,
        k: int = 5,
        regularizacion_cov: float = 1e-6,
    ) -> None:
        if estrategia not in ESTRATEGIAS:
            raise ValueError(f"estrategia debe ser una de {ESTRATEGIAS}, no {estrategia!r}")
        self.calibracion = calibracion
        self.estrategia = estrategia
        self.umbral_no_clasificable = umbral_no_clasificable
        self.k = k
        self.regularizacion_cov = regularizacion_cov

        self.etiquetas_: list[str] = calibracion.etiquetas
        self._medias = calibracion.matriz_centroides()  # (n_polimeros, d)

        d = calibracion.n_features
        self._cov_inv: dict[str, np.ndarray] = {}
        for etiqueta in self.etiquetas_:
            cov = np.atleast_2d(calibracion.covarianzas[etiqueta]).astype(float)
            cov = cov + np.eye(d) * regularizacion_cov
            self._cov_inv[etiqueta] = np.linalg.inv(cov)

        self._nn: NearestNeighbors | None = None
        self._nn_etiquetas: np.ndarray | None = None
        self._gmm: GaussianMixture | None = None

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
            return self  # todo lo necesario ya está en la calibración

        if self.estrategia == "knn":
            if X is None or y is None:
                raise ValueError("La estrategia 'knn' necesita X e y con las mediciones de calibración.")
            self._nn = NearestNeighbors(n_neighbors=min(self.k, len(X)))
            self._nn.fit(np.asarray(X, dtype=float))
            self._nn_etiquetas = np.asarray(y)
            return self

        # gmm: un componente por polímero, medias/precisión inicializadas con la calibración
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
            # Ajuste "de forma": muestreo sintético alrededor de cada centroide.
            muestras = np.vstack(
                [
                    np.random.default_rng(0).multivariate_normal(
                        self.calibracion.centroides[e], self.calibracion.covarianzas[e], size=200
                    )
                    for e in self.etiquetas_
                ]
            )
            self._gmm.fit(muestras)
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
            Score usado para la regla de "no clasificable" (distancia de Mahalanobis en σ,
            o distancia euclídea media según la estrategia). Útil para el reporte y para
            elegir el umbral.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.calibracion.n_features:
            raise ValueError(
                f"X debe tener forma (n, {self.calibracion.n_features}); recibí {X.shape}."
            )

        if self.estrategia == "knn":
            etiqueta_cruda, score = self._predecir_knn(X)
        else:  # centroide y gmm comparten la métrica de Mahalanobis al centroide
            etiqueta_cruda, score = self._predecir_mahalanobis(X)

        etiquetas = etiqueta_cruda.astype(object)
        if self.umbral_no_clasificable is not None:
            etiquetas[score > self.umbral_no_clasificable] = NO_CLASIFICABLE
        return etiquetas.astype(str), score

    def _predecir_mahalanobis(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Centroide más cercano por distancia de Mahalanobis (en unidades de σ)."""
        n = len(X)
        distancias = np.empty((n, len(self.etiquetas_)))
        for j, etiqueta in enumerate(self.etiquetas_):
            d2 = _mahalanobis_cuadrado(X, self._medias[j], self._cov_inv[etiqueta])
            distancias[:, j] = np.sqrt(np.maximum(d2, 0.0))
        j_min = distancias.argmin(axis=1)
        etiqueta = np.array(self.etiquetas_, dtype=object)[j_min]
        score = distancias[np.arange(n), j_min]
        return etiqueta, score

    def _predecir_knn(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Voto mayoritario de los k vecinos + distancia euclídea media como score."""
        if self._nn is None or self._nn_etiquetas is None:
            raise RuntimeError("Llamá a entrenar(X, y) antes de predecir con estrategia 'knn'.")
        dist, idx = self._nn.kneighbors(X)
        etiquetas_vecinos = self._nn_etiquetas[idx]  # (n, k)
        etiqueta = np.array(
            [_voto_mayoritario(fila) for fila in etiquetas_vecinos], dtype=object
        )
        score = dist.mean(axis=1)
        return etiqueta, score


def _voto_mayoritario(etiquetas: np.ndarray):
    """Etiqueta más frecuente en ``etiquetas`` (desempate por primera aparición)."""
    valores, cuentas = np.unique(etiquetas, return_counts=True)
    return valores[cuentas.argmax()]
