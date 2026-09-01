"""Reportes y visualización de resultados.

Estado: parcial. Hoy: :func:`resultados_a_dataframe` y :func:`guardar_reporte_metricas`.
Fase 3: diagrama de phasores con clusters de referencia + partículas clasificadas,
overlay de la imagen con ROIs etiquetadas, resumen estadístico por muestra.

Todo reporte de resultados incluye **siempre** las métricas estándar de
:mod:`napari_mp_classifier.metricas`, no solo el CSV de asignaciones
(requisito de documentación del proyecto).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .metricas import ReporteClasificacion


def resultados_a_dataframe(
    X: np.ndarray,
    etiquetas: np.ndarray,
    score: np.ndarray,
    columnas: list[str],
    ids: np.ndarray | None = None,
) -> pd.DataFrame:
    """Arma el CSV de asignaciones: una fila por partícula.

    Parameters
    ----------
    X : numpy.ndarray, shape (n, d)
        Coordenadas de phasor de cada partícula.
    etiquetas : array-like of str, shape (n,)
        Polímero asignado o ``"no_clasificable"``.
    score : array-like, shape (n,)
        Score de rechazo (distancia al cluster). Ver
        :meth:`ClasificadorPhasor.predecir_con_score`.
    columnas : list of str
        Nombres de las columnas de ``X``.
    ids : array-like, optional
        Identificador de cada partícula/ROI. Si es ``None`` se numera ``0..n-1``.

    Returns
    -------
    pandas.DataFrame
    """
    n = len(X)
    if ids is None:
        ids = np.arange(n)
    tabla = pd.DataFrame(X, columns=columnas)
    tabla.insert(0, "id", ids)
    tabla["polimero_predicho"] = np.asarray(etiquetas, dtype=str)
    tabla["score_rechazo"] = np.asarray(score, dtype=float)
    return tabla


def guardar_reporte_metricas(reporte: ReporteClasificacion, carpeta: str | Path) -> dict[str, Path]:
    """Escribe el reporte de métricas en disco (texto + CSVs).

    Genera ``metricas_resumen.txt``, ``metricas_por_clase.csv`` y
    ``matriz_confusion.csv`` en ``carpeta``.

    Returns
    -------
    dict[str, pathlib.Path]
        Rutas de los archivos escritos.
    """
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    rutas = {
        "resumen": carpeta / "metricas_resumen.txt",
        "por_clase": carpeta / "metricas_por_clase.csv",
        "matriz_confusion": carpeta / "matriz_confusion.csv",
    }
    rutas["resumen"].write_text(reporte.resumen(), encoding="utf-8")
    reporte.por_clase.to_csv(rutas["por_clase"])
    reporte.matriz_confusion.to_csv(rutas["matriz_confusion"])
    return rutas
