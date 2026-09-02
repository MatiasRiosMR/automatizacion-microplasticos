"""Pipeline completo: imagen de muestra → partículas clasificadas + reporte.

Encadena las etapas de las Fases 1–2 en una sola llamada:

``segmentar`` → ``extraer_features`` → (fusión FLIM+espectral implícita en el vector 4D)
→ ``ClasificadorPhasor`` → ``metricas``.

Es el punto de entrada de librería (:func:`analizar_muestra`) y lo que usa la CLI
(``napari-mp-classifier classify``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .calibracion import Calibracion
from .clasificador import ClasificadorPhasor
from .features import extraer_features, matriz_features
from .metricas import (
    ReporteClasificacion,
    ReporteSegmentacion,
    emparejar_rois,
    evaluar_clasificacion,
    evaluar_segmentacion,
)
from .segmentacion import segmentar

CANALES_PHASOR: tuple[str, ...] = ("g_flim", "s_flim", "g_esp", "s_esp")


@dataclass
class ResultadoMuestra:
    """Salida de :func:`analizar_muestra`.

    Attributes
    ----------
    features : pandas.DataFrame
        Una fila por ROI: features de forma + phasor + ``polimero_predicho`` +
        ``score_rechazo`` (+ ``polimero_real`` si se pasó verdad de terreno).
    labels : numpy.ndarray of int
        Segmentación de la muestra.
    columnas_phasor : list[str]
        Columnas de phasor usadas para clasificar (define la modalidad).
    calibracion : Calibracion
        Firma de referencia usada.
    reporte_clasificacion : ReporteClasificacion or None
        Métricas de clasificación (solo si se pasó ``verdad``).
    reporte_segmentacion : ReporteSegmentacion or None
        Métricas de segmentación (solo si se pasó ``verdad``).
    """

    features: pd.DataFrame
    labels: np.ndarray
    columnas_phasor: list[str]
    calibracion: Calibracion
    reporte_clasificacion: ReporteClasificacion | None = None
    reporte_segmentacion: ReporteSegmentacion | None = None
    parametros: dict = field(default_factory=dict)

    @property
    def n_rois(self) -> int:
        return len(self.features)

    def conteo_por_polimero(self) -> pd.Series:
        """Cantidad de ROIs asignadas a cada polímero (y a ``no_clasificable``)."""
        return self.features["polimero_predicho"].value_counts()


def analizar_muestra(
    canales: dict[str, np.ndarray],
    calibracion: Calibracion,
    *,
    estrategia: str = "knn",
    confianza: float | None = 0.99,
    metodo_segmentacion: str = "umbral",
    separar_contacto: bool = True,
    tam_min: int = 8,
    tam_max: int | None = None,
    escala_um_px: float | None = None,
    mediciones_calibracion: tuple[np.ndarray, np.ndarray] | None = None,
    mascara_celular: np.ndarray | None = None,
    verdad: dict | None = None,
    semilla: int = 0,
) -> ResultadoMuestra:
    """Analiza una imagen de muestra de punta a punta.

    Parameters
    ----------
    canales : dict[str, numpy.ndarray]
        Debe tener ``"intensidad"`` y al menos un par de phasor (``g_flim``/``s_flim``
        y/o ``g_esp``/``s_esp``), todos 2D y de la misma forma. La modalidad de
        clasificación se deduce de los pares presentes (fusión si están los cuatro).
    calibracion : Calibracion
        Firma de referencia. Su ``columnas`` debe ser coherente con los canales pasados.
    estrategia : {"knn", "centroide", "gmm"}, optional
        Estrategia del clasificador. Por defecto ``"knn"`` (más robusta para el rechazo,
        ver ``docs/RESULTADOS_FASE1.md``).
    confianza : float or None, optional
        Nivel de confianza de la regla ``"no_clasificable"``. Por defecto ``0.99``.
    metodo_segmentacion : {"umbral", "kmeans"}, optional
        Ver :func:`~napari_mp_classifier.segmentacion.segmentar`.
    separar_contacto : bool, optional
        Aplica ``watershed`` para separar partículas en contacto. Por defecto ``True``.
    tam_min, tam_max : int, optional
        Filtro de área de las ROIs, en px.
    escala_um_px : float or None, optional
        Tamaño de píxel en µm (para ``area_um2``).
    mediciones_calibracion : tuple(numpy.ndarray, numpy.ndarray), optional
        ``(X, y)`` con las mediciones individuales de calibración. **Obligatorio** si
        ``estrategia="knn"``.
    mascara_celular : numpy.ndarray, optional
        Si se pasa, solo se conservan las ROIs contenidas en ella (muestras de fagocitos,
        :func:`~napari_mp_classifier.segmentacion.restringir_a_mascara`).
    verdad : dict, optional
        ``{"labels": ..., "polimero": {label: codigo}}`` de verdad de terreno. Si se pasa,
        se calculan ``reporte_segmentacion`` y ``reporte_clasificacion``.
    semilla : int, optional

    Returns
    -------
    ResultadoMuestra

    Raises
    ------
    ValueError
        Si faltan canales, si la modalidad no coincide con la calibración, o si
        ``estrategia="knn"`` sin ``mediciones_calibracion``.
    """
    if "intensidad" not in canales:
        raise ValueError("canales debe incluir 'intensidad'.")
    tiene_flim = "g_flim" in canales and "s_flim" in canales
    tiene_esp = "g_esp" in canales and "s_esp" in canales
    if not (tiene_flim or tiene_esp):
        raise ValueError("canales debe incluir al menos g_flim/s_flim o g_esp/s_esp.")

    modalidad = "fusion" if (tiene_flim and tiene_esp) else ("flim" if tiene_flim else "espectral")
    columnas = {"flim": ["g_flim", "s_flim"], "espectral": ["g_esp", "s_esp"],
                "fusion": list(CANALES_PHASOR)}[modalidad]
    if list(calibracion.columnas) != columnas:
        raise ValueError(
            f"La calibración tiene columnas {list(calibracion.columnas)} pero la muestra "
            f"es modalidad '{modalidad}' ({columnas})."
        )

    labels = segmentar(
        canales["intensidad"],
        g_flim=canales.get("g_flim"), s_flim=canales.get("s_flim"),
        g_esp=canales.get("g_esp"), s_esp=canales.get("s_esp"),
        metodo=metodo_segmentacion, separar=separar_contacto,
        tam_min=tam_min, tam_max=tam_max, semilla=semilla,
    )
    if mascara_celular is not None:
        from .segmentacion import restringir_a_mascara

        labels = restringir_a_mascara(labels, mascara_celular)

    features = extraer_features(
        labels, canales["intensidad"],
        g_flim=canales.get("g_flim"), s_flim=canales.get("s_flim"),
        g_esp=canales.get("g_esp"), s_esp=canales.get("s_esp"),
        escala_um_px=escala_um_px,
    )

    clf = ClasificadorPhasor(calibracion, estrategia=estrategia, confianza=confianza)
    if estrategia == "knn":
        if mediciones_calibracion is None:
            raise ValueError("estrategia='knn' necesita mediciones_calibracion=(X, y).")
        clf.entrenar(*mediciones_calibracion)
    else:
        clf.entrenar()

    if len(features):
        X, _ = matriz_features(features, modalidad)
        pred, score = clf.predecir_con_score(X)
    else:
        pred, score = np.array([], dtype=str), np.array([], dtype=float)
    features = features.copy()
    features["polimero_predicho"] = pred
    features["score_rechazo"] = score

    reporte_seg = reporte_clf = None
    if verdad is not None:
        reporte_seg = evaluar_segmentacion(labels, verdad["labels"])
        emparejadas = emparejar_rois(labels, verdad["labels"], iou_min=0.3)
        verd_por_pred = {lp: verdad["polimero"][lv] for lv, (lp, _) in emparejadas.items()}
        from . import NO_CLASIFICABLE

        y_true = np.array(
            [verd_por_pred.get(label, NO_CLASIFICABLE) for label in features.index]
        )
        features["polimero_real"] = y_true
        if len(features):
            reporte_clf = evaluar_clasificacion(y_true, features["polimero_predicho"].to_numpy())

    return ResultadoMuestra(
        features=features,
        labels=labels,
        columnas_phasor=columnas,
        calibracion=calibracion,
        reporte_clasificacion=reporte_clf,
        reporte_segmentacion=reporte_seg,
        parametros={
            "modalidad": modalidad,
            "estrategia": estrategia,
            "confianza": confianza,
            "metodo_segmentacion": metodo_segmentacion,
            "escala_um_px": escala_um_px,
        },
    )
