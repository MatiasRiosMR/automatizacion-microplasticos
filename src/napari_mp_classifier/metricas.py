"""Métricas estándar de clasificación, comparables con la literatura.

Se reportan en el mismo formato que Meyers et al. (2022) y Ho et al. (2025) —FIMAP—
para que los resultados de este módulo sean directamente comparables:
exactitud, precisión, recall y F1 (macro y por polímero) + matriz de confusión.

La categoría ``"no_clasificable"`` se trata como una clase más en la matriz de confusión
(permite ver cuánta señal ambiental/celular se rechaza correctamente y cuánto polímero
real se pierde), pero se puede excluir de las métricas macro con ``incluir_no_clasificable``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from . import NO_CLASIFICABLE


@dataclass
class ReporteClasificacion:
    """Resultado de :func:`evaluar_clasificacion`.

    Attributes
    ----------
    exactitud : float
        Fracción de partículas correctamente clasificadas (accuracy global).
    precision_macro, recall_macro, f1_macro : float
        Promedio no ponderado sobre las clases consideradas.
    por_clase : pandas.DataFrame
        Precisión, recall, F1 y soporte por polímero.
    matriz_confusion : pandas.DataFrame
        Filas = verdad de terreno, columnas = predicción.
    etiquetas : list[str]
        Clases incluidas en la matriz de confusión.
    n : int
        Cantidad de partículas evaluadas.
    """

    exactitud: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    por_clase: pd.DataFrame
    matriz_confusion: pd.DataFrame
    etiquetas: list[str]
    n: int

    def resumen(self) -> str:
        """Texto legible con las métricas principales (para consola o log)."""
        lineas = [
            f"n = {self.n} partículas",
            f"exactitud       : {self.exactitud:.3f}",
            f"precisión (macro): {self.precision_macro:.3f}",
            f"recall (macro)  : {self.recall_macro:.3f}",
            f"F1 (macro)      : {self.f1_macro:.3f}",
            "",
            "Por polímero:",
            self.por_clase.to_string(),
            "",
            "Matriz de confusión (fila=real, columna=predicho):",
            self.matriz_confusion.to_string(),
        ]
        return "\n".join(lineas)


def evaluar_clasificacion(
    y_verdadero,
    y_predicho,
    etiquetas: list[str] | None = None,
    incluir_no_clasificable: bool = True,
) -> ReporteClasificacion:
    """Calcula las métricas estándar de clasificación.

    Parameters
    ----------
    y_verdadero, y_predicho : array-like of str
        Etiquetas reales y predichas (códigos de polímero o ``"no_clasificable"``).
    etiquetas : list of str, optional
        Orden explícito de clases para la matriz de confusión. Si es ``None`` se toman
        todas las clases presentes, ordenadas alfabéticamente con ``"no_clasificable"``
        al final.
    incluir_no_clasificable : bool, optional
        Si es ``False``, la clase ``"no_clasificable"`` se excluye de las métricas macro
        y por clase (sigue apareciendo en la matriz de confusión). Útil para comparar
        contra trabajos que solo reportan métricas de polímero.

    Returns
    -------
    ReporteClasificacion
    """
    y_verdadero = np.asarray(y_verdadero, dtype=object)
    y_predicho = np.asarray(y_predicho, dtype=object)
    if len(y_verdadero) != len(y_predicho):
        raise ValueError("y_verdadero e y_predicho deben tener la misma longitud.")

    if etiquetas is None:
        presentes = set(y_verdadero) | set(y_predicho)
        polimeros = sorted(p for p in presentes if p != NO_CLASIFICABLE)
        etiquetas = polimeros + ([NO_CLASIFICABLE] if NO_CLASIFICABLE in presentes else [])

    mc = confusion_matrix(y_verdadero, y_predicho, labels=etiquetas)
    matriz_confusion = pd.DataFrame(mc, index=etiquetas, columns=etiquetas)

    etiquetas_metricas = [
        e for e in etiquetas if incluir_no_clasificable or e != NO_CLASIFICABLE
    ]

    precision, recall, f1, soporte = precision_recall_fscore_support(
        y_verdadero,
        y_predicho,
        labels=etiquetas_metricas,
        zero_division=0,
    )
    por_clase = pd.DataFrame(
        {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "soporte": soporte,
        },
        index=etiquetas_metricas,
    )

    if incluir_no_clasificable:
        mascara = np.ones(len(y_verdadero), dtype=bool)
    else:
        mascara = y_verdadero != NO_CLASIFICABLE
    exactitud = (
        accuracy_score(y_verdadero[mascara], y_predicho[mascara]) if mascara.any() else float("nan")
    )

    return ReporteClasificacion(
        exactitud=float(exactitud),
        precision_macro=float(np.mean(precision)) if len(precision) else float("nan"),
        recall_macro=float(np.mean(recall)) if len(recall) else float("nan"),
        f1_macro=float(np.mean(f1)) if len(f1) else float("nan"),
        por_clase=por_clase,
        matriz_confusion=matriz_confusion,
        etiquetas=list(etiquetas),
        n=len(y_verdadero),
    )


# =========================================================================== segmentación


@dataclass
class ReporteSegmentacion:
    """Resultado de :func:`evaluar_segmentacion`.

    Attributes
    ----------
    iou_medio : float
        IoU promedio de las ROIs verdaderas emparejadas (0 si ninguna se detectó).
        Comparable con FIMAP (Ho et al. 2025: IoU 87,7 %).
    precision_deteccion : float
        Fracción de ROIs predichas que corresponden a una ROI verdadera (``1 - falsos
        positivos``).
    recall_deteccion : float
        Fracción de ROIs verdaderas que fueron detectadas.
    f1_deteccion : float
        Media armónica de precisión y recall de detección.
    n_verdaderas, n_predichas, n_emparejadas : int
        Conteos de ROIs.
    """

    iou_medio: float
    precision_deteccion: float
    recall_deteccion: float
    f1_deteccion: float
    n_verdaderas: int
    n_predichas: int
    n_emparejadas: int

    def resumen(self) -> str:
        """Texto legible con las métricas de segmentación."""
        return "\n".join(
            [
                f"ROIs verdaderas : {self.n_verdaderas}",
                f"ROIs predichas  : {self.n_predichas}",
                f"emparejadas     : {self.n_emparejadas}",
                f"IoU medio       : {self.iou_medio:.3f}",
                f"precisión det.  : {self.precision_deteccion:.3f}",
                f"recall det.     : {self.recall_deteccion:.3f}",
                f"F1 detección    : {self.f1_deteccion:.3f}",
            ]
        )


def emparejar_rois(
    labels_predicho: np.ndarray,
    labels_verdadero: np.ndarray,
    iou_min: float = 0.3,
) -> dict[int, tuple[int, float]]:
    """Empareja cada ROI verdadera con la ROI predicha de mayor solape (IoU).

    Parameters
    ----------
    labels_predicho, labels_verdadero : numpy.ndarray of int
        Imágenes de labels (``0`` = fondo).
    iou_min : float, optional
        IoU mínimo para aceptar un emparejamiento. Por defecto ``0.3``.

    Returns
    -------
    dict[int, tuple[int, float]]
        ``{label_verdadero: (label_predicho, iou)}`` solo para los pares con
        ``iou >= iou_min``.
    """
    pred = np.asarray(labels_predicho, dtype=int)
    verd = np.asarray(labels_verdadero, dtype=int)

    areas_pred = {int(i): int((pred == i).sum()) for i in np.unique(pred) if i > 0}
    emparejamiento: dict[int, tuple[int, float]] = {}
    for etiqueta_v in np.unique(verd):
        if etiqueta_v == 0:
            continue
        mascara_v = verd == etiqueta_v
        area_v = int(mascara_v.sum())
        solapados, cuentas = np.unique(pred[mascara_v], return_counts=True)
        mejor_iou, mejor_pred = 0.0, 0
        for etiqueta_p, interseccion in zip(solapados, cuentas):
            if etiqueta_p == 0:
                continue
            union = area_v + areas_pred[int(etiqueta_p)] - int(interseccion)
            iou = int(interseccion) / union if union else 0.0
            if iou > mejor_iou:
                mejor_iou, mejor_pred = iou, int(etiqueta_p)
        if mejor_iou >= iou_min:
            emparejamiento[int(etiqueta_v)] = (mejor_pred, mejor_iou)
    return emparejamiento


def evaluar_segmentacion(
    labels_predicho: np.ndarray,
    labels_verdadero: np.ndarray,
    iou_min: float = 0.5,
) -> ReporteSegmentacion:
    """Calcula IoU medio y precisión/recall de detección de una segmentación.

    Parameters
    ----------
    labels_predicho, labels_verdadero : numpy.ndarray of int
        Imágenes de labels (``0`` = fondo).
    iou_min : float, optional
        IoU mínimo para contar una ROI verdadera como detectada. Por defecto ``0.5``.

    Returns
    -------
    ReporteSegmentacion
    """
    n_verd = int(len(np.unique(labels_verdadero)) - (1 if 0 in labels_verdadero else 0))
    n_pred = int(len(np.unique(labels_predicho)) - (1 if 0 in labels_predicho else 0))

    emparejadas = emparejar_rois(labels_predicho, labels_verdadero, iou_min=iou_min)
    n_emp = len(emparejadas)
    ious = [iou for _, iou in emparejadas.values()]

    recall = n_emp / n_verd if n_verd else float("nan")
    precision = n_emp / n_pred if n_pred else float("nan")
    if precision and recall and np.isfinite(precision) and np.isfinite(recall):
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return ReporteSegmentacion(
        iou_medio=float(np.mean(ious)) if ious else 0.0,
        precision_deteccion=float(precision),
        recall_deteccion=float(recall),
        f1_deteccion=float(f1),
        n_verdaderas=n_verd,
        n_predichas=n_pred,
        n_emparejadas=n_emp,
    )
