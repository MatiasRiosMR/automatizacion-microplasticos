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
        n=int(len(y_verdadero)),
    )
