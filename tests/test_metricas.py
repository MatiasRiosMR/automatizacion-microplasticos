"""Tests de :mod:`napari_mp_classifier.metricas`."""

import numpy as np

from napari_mp_classifier import NO_CLASIFICABLE
from napari_mp_classifier.metricas import evaluar_clasificacion


def test_clasificacion_perfecta():
    y = np.array(["PET", "PET", "PS", "PVC", "PP"])
    rep = evaluar_clasificacion(y, y)
    assert rep.exactitud == 1.0
    assert rep.f1_macro == 1.0
    assert rep.n == 5
    assert np.trace(rep.matriz_confusion.to_numpy()) == 5


def test_matriz_de_confusion_orientacion():
    y_true = np.array(["PET", "PET", "PS"])
    y_pred = np.array(["PET", "PS", "PS"])
    rep = evaluar_clasificacion(y_true, y_pred, etiquetas=["PET", "PS"])
    # fila = real, columna = predicho -> 1 PET clasificado como PS
    assert rep.matriz_confusion.loc["PET", "PS"] == 1
    assert rep.matriz_confusion.loc["PET", "PET"] == 1
    assert rep.matriz_confusion.loc["PS", "PS"] == 1


def test_excluir_no_clasificable_de_metricas_macro():
    y_true = np.array(["PET", "PS", NO_CLASIFICABLE, NO_CLASIFICABLE])
    y_pred = np.array(["PET", "PS", NO_CLASIFICABLE, "PET"])

    con = evaluar_clasificacion(y_true, y_pred, incluir_no_clasificable=True)
    sin = evaluar_clasificacion(y_true, y_pred, incluir_no_clasificable=False)

    assert NO_CLASIFICABLE in con.por_clase.index
    assert NO_CLASIFICABLE not in sin.por_clase.index
    # la matriz de confusión mantiene la clase en ambos casos
    assert NO_CLASIFICABLE in sin.matriz_confusion.index


def test_resumen_es_texto():
    y = np.array(["PET", "PS", "PP", "PVC"])
    texto = evaluar_clasificacion(y, y).resumen()
    assert "exactitud" in texto
    assert "Matriz de confusión" in texto
