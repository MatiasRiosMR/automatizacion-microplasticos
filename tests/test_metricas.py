"""Tests de :mod:`napari_mp_classifier.metricas`."""

import numpy as np
import pytest

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


def test_exactitud_balanceada_no_se_infla_con_clase_mayoritaria():
    y = np.array(["PET"] * 90 + ["PS"] * 10)
    p = np.array(["PET"] * 100)
    rep = evaluar_clasificacion(y, p)
    assert rep.exactitud == pytest.approx(0.9)
    assert rep.exactitud_balanceada == pytest.approx(0.5)


def test_emparejar_rois_es_uno_a_uno():
    from napari_mp_classifier.metricas import emparejar_rois

    verdad = np.zeros((10, 20), dtype=int)
    verdad[2:8, 2:9] = 1
    verdad[2:8, 11:18] = 2
    pred = np.zeros_like(verdad)
    pred[2:8, 2:18] = 1  # una sola ROI predicha cubre las dos partículas
    emp = emparejar_rois(pred, verdad, iou_min=0.3)
    assert len(emp) == 1  # antes: las dos verdaderas apuntaban a la misma predicha
