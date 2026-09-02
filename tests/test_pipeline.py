"""Tests de :mod:`napari_mp_classifier.pipeline`."""

import numpy as np
import pytest
from datos_sinteticos import _columnas, generar_calibracion, generar_imagen_muestra

from napari_mp_classifier import Calibracion, analizar_muestra


def _calibracion(modalidad="fusion", semilla=0):
    df = generar_calibracion(modalidad, n_por_polimero=60, semilla=semilla)
    cal = Calibracion.desde_dataframe(df, columnas=_columnas(modalidad))
    mediciones = (df[_columnas(modalidad)].to_numpy(), df["polimero"].to_numpy())
    return cal, mediciones


@pytest.fixture
def muestra():
    return generar_imagen_muestra(semilla=4)


def test_analizar_muestra_fusion_end_to_end(muestra):
    canales, verdad = muestra
    cal, mediciones = _calibracion("fusion")
    res = analizar_muestra(
        canales, cal, estrategia="knn", mediciones_calibracion=mediciones,
        escala_um_px=0.18, verdad=verdad,
    )
    assert res.n_rois > 10
    assert res.parametros["modalidad"] == "fusion"
    assert {"polimero_predicho", "score_rechazo", "polimero_real"}.issubset(res.features.columns)
    assert res.reporte_segmentacion.iou_medio > 0.7
    # El clasificador de la Fase 1 se traslada: exactitud alta sobre ROIs de polímero.
    solo_pol = res.features["polimero_real"] != "no_clasificable"
    aciertos = (
        res.features.loc[solo_pol, "polimero_predicho"]
        == res.features.loc[solo_pol, "polimero_real"]
    )
    assert aciertos.mean() > 0.85


def test_modalidad_se_deduce_de_los_canales(muestra):
    canales, _ = muestra
    solo_esp = {k: canales[k] for k in ("intensidad", "g_esp", "s_esp")}
    cal, _ = _calibracion("espectral")
    res = analizar_muestra(solo_esp, cal, estrategia="centroide")
    assert res.parametros["modalidad"] == "espectral"
    assert res.columnas_phasor == ["g_esp", "s_esp"]


def test_calibracion_incoherente_con_la_muestra(muestra):
    canales, _ = muestra
    cal_flim, _ = _calibracion("flim")
    with pytest.raises(ValueError, match="modalidad"):
        analizar_muestra(canales, cal_flim, estrategia="centroide")


def test_knn_sin_mediciones_falla(muestra):
    canales, _ = muestra
    cal, _ = _calibracion("fusion")
    with pytest.raises(ValueError, match="mediciones_calibracion"):
        analizar_muestra(canales, cal, estrategia="knn")


def test_falta_intensidad():
    cal, _ = _calibracion("fusion")
    with pytest.raises(ValueError, match="intensidad"):
        analizar_muestra({"g_flim": np.zeros((4, 4))}, cal)


def test_mascara_celular_filtra_rois(muestra):
    canales, _ = muestra
    cal, mediciones = _calibracion("fusion")
    mascara = np.zeros(canales["intensidad"].shape, dtype=bool)
    mascara[: mascara.shape[0] // 2] = True  # media imagen

    completo = analizar_muestra(canales, cal, estrategia="knn", mediciones_calibracion=mediciones)
    filtrado = analizar_muestra(
        canales, cal, estrategia="knn", mediciones_calibracion=mediciones,
        mascara_celular=mascara,
    )
    assert filtrado.n_rois < completo.n_rois
    assert (filtrado.features["centro_fila"] < mascara.shape[0] // 2 + 5).all()


def test_conteo_por_polimero(muestra):
    canales, _ = muestra
    cal, mediciones = _calibracion("fusion")
    res = analizar_muestra(canales, cal, estrategia="knn", mediciones_calibracion=mediciones)
    conteo = res.conteo_por_polimero()
    assert conteo.sum() == res.n_rois
