"""Tests de :mod:`napari_mp_classifier.reportes`."""

import numpy as np
import pytest

from napari_mp_classifier import Calibracion, ClasificadorPhasor
from napari_mp_classifier.metricas import evaluar_clasificacion
from napari_mp_classifier.reportes import (
    figura_comparacion,
    figura_matriz_confusion,
    figura_metricas_por_clase,
    figura_phasores,
    guardar_figura,
    guardar_reporte_metricas,
    resultados_a_dataframe,
)
from datos_sinteticos import generar_calibracion, generar_particulas


def _caso(modalidad, columnas):
    df = generar_calibracion(modalidad, n_por_polimero=40)
    cal = Calibracion.desde_dataframe(df, columnas=columnas)
    clf = ClasificadorPhasor(cal, estrategia="centroide").entrenar()
    X, y = generar_particulas(modalidad, n_por_polimero=15, n_no_clasificables=20)
    pred = clf.predecir(X)
    rep = evaluar_clasificacion(y, pred)
    return cal, X, y, pred, rep


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


@pytest.mark.parametrize(
    "modalidad,columnas",
    [
        ("flim", ["g_flim", "s_flim"]),
        ("espectral", ["g_esp", "s_esp"]),
        ("fusion", ["g_flim", "s_flim", "g_esp", "s_esp"]),
    ],
)
def test_figura_phasores_por_modalidad(modalidad, columnas):
    cal, X, y, pred, _ = _caso(modalidad, columnas)
    fig = figura_phasores(cal, X, pred, columnas, etiquetas_reales=y, resaltar_errores=True)
    n_paneles = 2 if modalidad == "fusion" else 1
    assert len(fig.axes) == n_paneles
    for ax in fig.axes:
        assert ax.get_xlim()[0] < ax.get_xlim()[1]


def test_figura_phasores_submuestrea():
    columnas = ["g_flim", "s_flim"]
    cal, X, _y, pred, _ = _caso("flim", columnas)
    fig = figura_phasores(cal, X, pred, columnas, max_puntos=10)
    graficados = sum(col.get_offsets().shape[0] for col in fig.axes[0].collections)
    # 10 partículas submuestreadas + 6 centroides de referencia.
    assert graficados <= 10 + len(cal.etiquetas)


def test_figuras_metricas_y_guardado(tmp_path):
    _, _, _, _, rep = _caso("fusion", ["g_flim", "s_flim", "g_esp", "s_esp"])

    fig_mc = figura_matriz_confusion(rep)
    fig_pc = figura_metricas_por_clase(rep)
    tabla = [
        {"modalidad": "flim", "estrategia": "centroide", "exactitud": 0.88, "F1_polimeros": 0.87},
        {"modalidad": "fusion", "estrategia": "knn", "exactitud": 0.99, "F1_polimeros": 0.99},
    ]
    fig_cmp = figura_comparacion(tabla)

    for nombre, fig in [("mc", fig_mc), ("pc", fig_pc), ("cmp", fig_cmp)]:
        rutas = guardar_figura(fig, tmp_path / nombre, formatos=("png", "pdf"))
        assert [r.suffix for r in rutas] == [".png", ".pdf"]
        assert all(r.exists() and r.stat().st_size > 0 for r in rutas)


def test_figura_matriz_confusion_normalizar_invalido():
    _, _, _, _, rep = _caso("flim", ["g_flim", "s_flim"])
    with pytest.raises(ValueError):
        figura_matriz_confusion(rep, normalizar="columna")
