"""Fusión de las modalidades de phasor FLIM + espectral por partícula.

Núcleo diferenciador del proyecto: ningún antecedente (Sancataldo 2020, Meyers 2022,
FIMAP 2025, Rermborirak 2025) combina ambas modalidades. Ver ``docs/ANTECEDENTES.md``.

Tres situaciones, según cómo se adquieran los datos (pregunta 9 de
``docs/PREGUNTAS_DATOS.md``):

1. **Una sola imagen registrada** con los cuatro canales de phasor por píxel
   (``g_flim``, ``s_flim``, ``g_esp``, ``s_esp``): no hace falta este módulo — se llama
   :func:`~napari_mp_classifier.features.extraer_features` una vez con los cuatro canales
   y :func:`~napari_mp_classifier.features.matriz_features` con ``modalidad="fusion"``.

2. **Dos imágenes registradas espacialmente** pero segmentadas por separado (p. ej. la
   ``.sdt`` y la ``.czi`` tienen distinta resolución): :func:`fusionar_por_roi` empareja
   las ROIs por cercanía de centroide y concatena sus phasores en un vector de 4D.

3. **Adquisiciones independientes / no registrables**: :func:`fusionar_por_decision`
   clasifica cada modalidad por separado y combina las decisiones (producto de expertos
   simplificado): si coinciden, ese polímero; si discrepan o alguna rechaza, la partícula
   queda ``"no_clasificable"``.

La calibración y las features 4D ya están soportadas por
:class:`~napari_mp_classifier.calibracion.Calibracion` y
:class:`~napari_mp_classifier.clasificador.ClasificadorPhasor`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import NO_CLASIFICABLE

COLUMNAS_FUSION: tuple[str, ...] = ("g_flim", "s_flim", "g_esp", "s_esp")


def fusionar_por_roi(
    features_flim: pd.DataFrame,
    features_esp: pd.DataFrame,
    *,
    tol_centro_px: float = 8.0,
    sufijos: tuple[str, str] = ("_flim", "_esp"),
) -> pd.DataFrame:
    """Empareja ROIs de dos segmentaciones registradas y fusiona sus phasores.

    Cada ROI de ``features_flim`` se empareja con la ROI de ``features_esp`` de centroide
    más cercano, si la distancia es menor a ``tol_centro_px``. Las ROIs sin pareja se
    descartan (no hay información de las dos modalidades).

    Parameters
    ----------
    features_flim, features_esp : pandas.DataFrame
        Salidas de :func:`~napari_mp_classifier.features.extraer_features`, una con
        ``g_flim``/``s_flim`` y otra con ``g_esp``/``s_esp``. Ambas deben tener
        ``centro_fila`` y ``centro_col`` en el **mismo sistema de coordenadas**.
    tol_centro_px : float, optional
        Distancia máxima entre centroides para aceptar el emparejamiento. Por defecto ``8``.
    sufijos : tuple of str, optional
        Sufijos para las columnas de forma que existan en ambas tablas. Por defecto
        ``("_flim", "_esp")``.

    Returns
    -------
    pandas.DataFrame
        Una fila por par de ROIs emparejadas. Índice ``RangeIndex``. Columnas:
        ``label_flim``, ``label_esp``, ``dist_centro_px``, las cuatro de
        :data:`COLUMNAS_FUSION`, ``dispersion_flim``/``dispersion_esp`` si estaban, y las
        features de forma de cada modalidad con su sufijo.

    Raises
    ------
    ValueError
        Si falta ``g_flim``/``s_flim`` en ``features_flim`` o ``g_esp``/``s_esp`` en
        ``features_esp``, o las columnas de centroide.
    """
    _exigir_columnas(features_flim, ["g_flim", "s_flim", "centro_fila", "centro_col"], "features_flim")
    _exigir_columnas(features_esp, ["g_esp", "s_esp", "centro_fila", "centro_col"], "features_esp")

    centros_esp = features_esp[["centro_fila", "centro_col"]].to_numpy(dtype=float)
    usados: set[int] = set()
    filas: list[dict] = []

    for label_flim, fila_flim in features_flim.iterrows():
        centro = fila_flim[["centro_fila", "centro_col"]].to_numpy(dtype=float)
        distancias = np.linalg.norm(centros_esp - centro, axis=1)
        for idx in np.argsort(distancias):
            if idx in usados:
                continue
            if distancias[idx] > tol_centro_px:
                break
            fila_esp = features_esp.iloc[idx]
            usados.add(int(idx))
            filas.append(
                _fila_fusionada(label_flim, fila_flim, fila_esp, features_esp.index[idx],
                                float(distancias[idx]), sufijos)
            )
            break

    return pd.DataFrame(filas)


def _fila_fusionada(label_flim, fila_flim, fila_esp, label_esp, distancia, sufijos):
    fila: dict = {
        "label_flim": label_flim,
        "label_esp": label_esp,
        "dist_centro_px": distancia,
        "g_flim": float(fila_flim["g_flim"]),
        "s_flim": float(fila_flim["s_flim"]),
        "g_esp": float(fila_esp["g_esp"]),
        "s_esp": float(fila_esp["s_esp"]),
    }
    for col in ("dispersion_flim",):
        if col in fila_flim:
            fila[col] = float(fila_flim[col])
    for col in ("dispersion_esp",):
        if col in fila_esp:
            fila[col] = float(fila_esp[col])

    comunes = {"area_px", "area_um2", "intensidad_total", "intensidad_media",
               "excentricidad", "solidez", "extension", "relacion_aspecto", "perimetro"}
    for col in comunes:
        if col in fila_flim:
            fila[f"{col}{sufijos[0]}"] = float(fila_flim[col])
        if col in fila_esp:
            fila[f"{col}{sufijos[1]}"] = float(fila_esp[col])
    return fila


def fusionar_por_decision(
    pred_flim,
    pred_esp,
    *,
    score_flim=None,
    score_esp=None,
    umbral_score: float = 1.0,
) -> np.ndarray:
    """Combina dos clasificaciones independientes (una por modalidad).

    Regla (producto de expertos simplificado, conservador para falsos positivos):

    - Ambas modalidades asignan el **mismo polímero** → ese polímero.
    - Discrepan → ``"no_clasificable"``.
    - Alguna ya devolvió ``"no_clasificable"``, o su score supera ``umbral_score`` →
      ``"no_clasificable"``.

    Parameters
    ----------
    pred_flim, pred_esp : array-like of str, shape (n,)
        Predicción de cada modalidad para las **mismas** partículas, en el mismo orden.
    score_flim, score_esp : array-like, shape (n,), optional
        Score de rechazo de cada modalidad (cociente score/umbral de
        :meth:`~napari_mp_classifier.clasificador.ClasificadorPhasor.predecir_con_score`).
        Si se pasan, una partícula con score alto en cualquiera de las dos se rechaza.
    umbral_score : float, optional
        Umbral sobre el score normalizado. Por defecto ``1.0``.

    Returns
    -------
    numpy.ndarray of str, shape (n,)
        Predicción combinada.

    Raises
    ------
    ValueError
        Si ``pred_flim`` y ``pred_esp`` no tienen la misma longitud.
    """
    pred_flim = np.asarray(pred_flim, dtype=object)
    pred_esp = np.asarray(pred_esp, dtype=object)
    if len(pred_flim) != len(pred_esp):
        raise ValueError("pred_flim y pred_esp deben tener la misma longitud.")

    combinada = np.where(pred_flim == pred_esp, pred_flim, NO_CLASIFICABLE).astype(object)
    combinada[(pred_flim == NO_CLASIFICABLE) | (pred_esp == NO_CLASIFICABLE)] = NO_CLASIFICABLE

    for score in (score_flim, score_esp):
        if score is not None:
            combinada[np.asarray(score, dtype=float) > umbral_score] = NO_CLASIFICABLE

    return combinada.astype(str)


def _exigir_columnas(df: pd.DataFrame, columnas: list[str], nombre: str) -> None:
    faltantes = [c for c in columnas if c not in df.columns]
    if faltantes:
        raise ValueError(f"{nombre} no tiene las columnas {faltantes}.")
