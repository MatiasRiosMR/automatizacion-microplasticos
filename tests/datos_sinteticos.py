"""Generación de datos sintéticos de phasores para tests y desarrollo (Fase 1).

No pretende reproducir valores físicos reales: genera 6 clusters separables con dispersión
y solapamiento realistas, más una población de "materia orgánica / autofluorescencia" que
debe caer como ``"no_clasificable"``. Los centroides FLIM se derivan de tiempos de vida
plausibles mediante :func:`phasorpy.lifetime.phasor_from_lifetime`; los espectrales se
colocan a mano sobre el arco del plano de phasores.

Cuando el equipo entregue ``.sdt``/``.czi`` reales, estos generadores se reemplazan por
la calibración medida; los tests de ``clasificador`` y ``metricas`` siguen valiendo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from phasorpy.lifetime import phasor_from_lifetime

from napari_mp_classifier import POLIMEROS

# Tiempos de vida (ns) ILUSTRATIVOS de Nile Red en cada polímero, ordenados como POLIMEROS.
# Elegidos solo para dar 6 clusters FLIM separables; NO son valores medidos.
LIFETIMES_ILUSTRATIVOS_NS: dict[str, float] = {
    "PET": 2.1,
    "HDPE": 3.9,
    "PVC": 1.6,
    "LDPE": 3.4,
    "PP": 4.6,
    "PS": 2.8,
}

# Posiciones espectrales ILUSTRATIVAS (g_esp, s_esp) sobre el arco de phasores.
POSICIONES_ESPECTRALES_ILUSTRATIVAS: dict[str, tuple[float, float]] = {
    "PET": (0.55, 0.50),
    "HDPE": (0.10, 0.30),
    "PVC": (0.70, 0.44),
    "LDPE": (0.20, 0.40),
    "PP": (0.02, 0.15),
    "PS": (0.40, 0.49),
}

FRECUENCIA_FLIM_MHZ = 80.0


def centroides_referencia(modalidad: str = "flim") -> dict[str, np.ndarray]:
    """Centroides teóricos de los 6 polímeros para la modalidad pedida.

    Parameters
    ----------
    modalidad : {"flim", "espectral", "fusion"}

    Returns
    -------
    dict[str, numpy.ndarray]
        Centroide por polímero: 2D para ``flim``/``espectral``, 4D para ``fusion``
        (orden: ``g_flim, s_flim, g_esp, s_esp``).
    """
    g_flim, s_flim = phasor_from_lifetime(
        FRECUENCIA_FLIM_MHZ, [LIFETIMES_ILUSTRATIVOS_NS[p] for p in POLIMEROS]
    )
    flim = {p: np.array([g_flim[i], s_flim[i]]) for i, p in enumerate(POLIMEROS)}
    esp = {p: np.array(POSICIONES_ESPECTRALES_ILUSTRATIVAS[p]) for p in POLIMEROS}

    if modalidad == "flim":
        return flim
    if modalidad == "espectral":
        return esp
    if modalidad == "fusion":
        return {p: np.concatenate([flim[p], esp[p]]) for p in POLIMEROS}
    raise ValueError(f"modalidad desconocida: {modalidad!r}")


def _cov_isotropica(d: int, sigma: float) -> np.ndarray:
    return np.eye(d) * sigma**2


def generar_calibracion(
    modalidad: str = "flim",
    n_por_polimero: int = 60,
    sigma: float = 0.02,
    semilla: int = 0,
) -> pd.DataFrame:
    """Genera mediciones de calibración sintéticas (una fila por medición).

    Parameters
    ----------
    modalidad : {"flim", "espectral", "fusion"}
    n_por_polimero : int
        Mediciones simuladas por polímero.
    sigma : float
        Desvío estándar del ruido gaussiano isotrópico en el plano de phasores.
    semilla : int

    Returns
    -------
    pandas.DataFrame
        Columnas según modalidad + columna ``polimero``.
    """
    rng = np.random.default_rng(semilla)
    centroides = centroides_referencia(modalidad)
    columnas = _columnas(modalidad)
    d = len(columnas)

    filas = []
    for p, centro in centroides.items():
        muestras = rng.multivariate_normal(centro, _cov_isotropica(d, sigma), size=n_por_polimero)
        for fila in muestras:
            filas.append({**dict(zip(columnas, fila)), "polimero": p})
    return pd.DataFrame(filas)


def generar_particulas(
    modalidad: str = "flim",
    n_por_polimero: int = 40,
    n_no_clasificables: int = 60,
    sigma: float = 0.025,
    semilla: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Genera partículas a clasificar con verdad de terreno.

    Incluye ``n_no_clasificables`` puntos de "materia orgánica / autofluorescencia":
    una nube ancha y desplazada del conjunto de clusters, etiquetada ``"no_clasificable"``.

    Parameters
    ----------
    modalidad : {"flim", "espectral", "fusion"}
    n_por_polimero : int
    n_no_clasificables : int
    sigma : float
    semilla : int

    Returns
    -------
    X : numpy.ndarray, shape (n, d)
    y : numpy.ndarray of str, shape (n,)
    """
    rng = np.random.default_rng(semilla)
    centroides = centroides_referencia(modalidad)
    columnas = _columnas(modalidad)
    d = len(columnas)

    X_list, y_list = [], []
    for p, centro in centroides.items():
        X_list.append(rng.multivariate_normal(centro, _cov_isotropica(d, sigma), size=n_por_polimero))
        y_list.append(np.full(n_por_polimero, p))

    if n_no_clasificables > 0:
        centro_organico = np.mean(list(centroides.values()), axis=0) + 0.22
        cov_organico = _cov_isotropica(d, sigma * 4)
        X_list.append(rng.multivariate_normal(centro_organico, cov_organico, size=n_no_clasificables))
        y_list.append(np.full(n_no_clasificables, "no_clasificable"))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    orden = rng.permutation(len(X))
    return X[orden], y[orden]


def _columnas(modalidad: str) -> list[str]:
    return {
        "flim": ["g_flim", "s_flim"],
        "espectral": ["g_esp", "s_esp"],
        "fusion": ["g_flim", "s_flim", "g_esp", "s_esp"],
    }[modalidad]
