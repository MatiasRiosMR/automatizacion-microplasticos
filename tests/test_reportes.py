"""Tests de :mod:`napari_mp_classifier.reportes`."""

import numpy as np

from napari_mp_classifier import Calibracion, ClasificadorPhasor
from napari_mp_classifier.metricas import evaluar_clasificacion
from napari_mp_classifier.reportes import guardar_reporte_metricas, resultados_a_dataframe
from datos_sinteticos import generar_calibracion, generar_particulas


def test_resultados_a_dataframe():
    df = generar_calibracion("flim", n_por_polimero=60)
    cal = Calibracion.desde_dataframe(df, columnas=["g_flim", "s_flim"])
    clf = ClasificadorPhasor(cal).entrenar()

    X, _ = generar_particulas("flim", n_por_polimero=10, n_no_clasificables=10)
    etiquetas, score = clf.predecir_con_score(X)
    tabla = resultados_a_dataframe(X, etiquetas, score, columnas=["g_flim", "s_flim"])

    assert list(tabla.columns) == ["id", "g_flim", "s_flim", "polimero_predicho", "score_rechazo"]
    assert len(tabla) == len(X)
    assert tabla["id"].tolist() == list(range(len(X)))


def test_guardar_reporte_metricas(tmp_path):
    y = np.array(["PET", "PS", "PP", "PVC", "no_clasificable"])
    rep = evaluar_clasificacion(y, y)
    rutas = guardar_reporte_metricas(rep, tmp_path / "salida")

    for ruta in rutas.values():
        assert ruta.exists()
    assert "exactitud" in rutas["resumen"].read_text(encoding="utf-8")
