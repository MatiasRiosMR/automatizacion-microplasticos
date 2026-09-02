"""Tests de :mod:`napari_mp_classifier.features`."""

import numpy as np
import pytest
from datos_sinteticos import (
    _columnas,
    centroides_referencia,
    generar_calibracion,
    generar_imagen_muestra,
)

from napari_mp_classifier import Calibracion, ClasificadorPhasor
from napari_mp_classifier.features import COLUMNAS_FORMA, extraer_features, matriz_features
from napari_mp_classifier.metricas import emparejar_rois, evaluar_clasificacion
from napari_mp_classifier.segmentacion import segmentar


@pytest.fixture
def muestra_segmentada():
    canales, verdad = generar_imagen_muestra(semilla=1)
    labels = segmentar(
        canales["intensidad"],
        g_flim=canales["g_flim"],
        s_flim=canales["s_flim"],
        g_esp=canales["g_esp"],
        s_esp=canales["s_esp"],
        metodo="umbral",
    )
    return canales, verdad, labels


def test_columnas_y_una_fila_por_roi(muestra_segmentada):
    canales, _, labels = muestra_segmentada
    feats = extraer_features(
        labels,
        canales["intensidad"],
        g_flim=canales["g_flim"],
        s_flim=canales["s_flim"],
        g_esp=canales["g_esp"],
        s_esp=canales["s_esp"],
        escala_um_px=0.2,
    )
    assert len(feats) == labels.max()
    assert list(feats.index) == list(range(1, labels.max() + 1))
    for col in COLUMNAS_FORMA:
        assert col in feats.columns
    for sufijo in ("flim", "esp"):
        assert {f"g_{sufijo}", f"s_{sufijo}", f"dispersion_{sufijo}"}.issubset(feats.columns)
    assert np.allclose(feats["area_um2"], feats["area_px"] * 0.04)


def test_area_um2_nan_sin_escala(muestra_segmentada):
    canales, _, labels = muestra_segmentada
    feats = extraer_features(labels, canales["intensidad"], g_flim=canales["g_flim"], s_flim=canales["s_flim"])
    assert feats["area_um2"].isna().all()


def test_solo_una_modalidad(muestra_segmentada):
    canales, _, labels = muestra_segmentada
    feats = extraer_features(labels, canales["intensidad"], g_esp=canales["g_esp"], s_esp=canales["s_esp"])
    assert "g_esp" in feats.columns
    assert "g_flim" not in feats.columns


def test_g_sin_s_falla(muestra_segmentada):
    canales, _, labels = muestra_segmentada
    with pytest.raises(ValueError, match="g y s"):
        extraer_features(labels, canales["intensidad"], g_flim=canales["g_flim"])


def test_phasor_por_roi_cerca_del_centroide_verdadero(muestra_segmentada):
    canales, verdad, labels = muestra_segmentada
    feats = extraer_features(
        labels,
        canales["intensidad"],
        g_flim=canales["g_flim"],
        s_flim=canales["s_flim"],
        g_esp=canales["g_esp"],
        s_esp=canales["s_esp"],
    )
    emparejadas = emparejar_rois(labels, verdad["labels"], iou_min=0.5)
    centros = centroides_referencia("fusion")

    comparadas = 0
    for label_verdad, (label_pred, _) in emparejadas.items():
        codigo = verdad["polimero"][label_verdad]
        if codigo == "no_clasificable" or label_pred not in feats.index:
            continue
        fila = feats.loc[label_pred]
        estimado = np.array([fila["g_flim"], fila["s_flim"], fila["g_esp"], fila["s_esp"]])
        assert np.linalg.norm(estimado - centros[codigo]) < 0.12
        comparadas += 1
    assert comparadas >= 5


def test_matriz_features_orden_y_errores(muestra_segmentada):
    canales, _, labels = muestra_segmentada
    feats = extraer_features(
        labels,
        canales["intensidad"],
        g_flim=canales["g_flim"],
        s_flim=canales["s_flim"],
        g_esp=canales["g_esp"],
        s_esp=canales["s_esp"],
    )
    X, columnas = matriz_features(feats, "fusion")
    assert columnas == ["g_flim", "s_flim", "g_esp", "s_esp"]
    assert X.shape == (len(feats), 4)

    with pytest.raises(ValueError, match="modalidad"):
        matriz_features(feats, "raro")

    solo_esp = extraer_features(labels, canales["intensidad"], g_esp=canales["g_esp"], s_esp=canales["s_esp"])
    with pytest.raises(ValueError, match="columnas"):
        matriz_features(solo_esp, "fusion")


def test_pipeline_seg_features_clasificacion():
    """Seg -> features -> clasificador: las ROIs bien segmentadas se clasifican bien."""
    canales, verdad = generar_imagen_muestra(semilla=2)
    labels = segmentar(
        canales["intensidad"],
        g_flim=canales["g_flim"],
        s_flim=canales["s_flim"],
        g_esp=canales["g_esp"],
        s_esp=canales["s_esp"],
        metodo="umbral",
    )
    feats = extraer_features(
        labels,
        canales["intensidad"],
        g_flim=canales["g_flim"],
        s_flim=canales["s_flim"],
        g_esp=canales["g_esp"],
        s_esp=canales["s_esp"],
    )
    df_cal = generar_calibracion("fusion", n_por_polimero=60, semilla=0)
    cal = Calibracion.desde_dataframe(df_cal, columnas=_columnas("fusion"))
    clf = ClasificadorPhasor(cal, estrategia="knn", confianza=0.99)
    clf.entrenar(df_cal[_columnas("fusion")].to_numpy(), df_cal["polimero"].to_numpy())

    X, _ = matriz_features(feats, "fusion")
    pred = clf.predecir(X)

    emparejadas = emparejar_rois(labels, verdad["labels"], iou_min=0.3)
    verd_por_pred = {lp: verdad["polimero"][lv] for lv, (lp, _) in emparejadas.items()}
    y_true = np.array([verd_por_pred.get(l, "no_clasificable") for l in feats.index])

    solo_polimero = y_true != "no_clasificable"
    rep = evaluar_clasificacion(y_true[solo_polimero], pred[solo_polimero])
    assert rep.exactitud > 0.85
