"""Tests de :mod:`napari_mp_classifier.informe_html` y :mod:`napari_mp_classifier.glosario`."""

import re

import numpy as np
import pytest
from datos_sinteticos import _columnas, generar_calibracion, generar_imagen_muestra

from napari_mp_classifier import Calibracion, analizar_muestra
from napari_mp_classifier.glosario import GLOSARIO, ayuda
from napari_mp_classifier.informe_html import alertas, exportar_a_pdf, generar_informe_html


@pytest.fixture(scope="module")
def resultado_con_verdad():
    canales, verdad = generar_imagen_muestra(semilla=11)
    df = generar_calibracion("fusion", n_por_polimero=60, semilla=0)
    cols = _columnas("fusion")
    cal = Calibracion.desde_dataframe(df, columnas=cols)
    res = analizar_muestra(canales, cal, estrategia="knn", escala_um_px=0.18, verdad=verdad,
                           mediciones_calibracion=(df[cols].to_numpy(), df["polimero"].to_numpy()))
    return res, canales


def test_informe_con_verdad_tiene_todas_las_secciones(resultado_con_verdad, tmp_path):
    res, canales = resultado_con_verdad
    ruta = generar_informe_html(res, canales, tmp_path / "inf.html", nombre_muestra="M-01",
                                descripcion="control", exportar_pdf=False)
    html = ruta.read_text(encoding="utf-8")
    for texto in ("Muestra y condiciones del análisis", "Composición", "Firma de fluorescencia",
                  "Partículas con observaciones", "Resultados por partícula",
                  "Verificación con composición conocida", "Observaciones", "Método",
                  "Alcance y limitaciones", "Calibración de referencia", "Glosario"):
        assert texto in html, texto
    assert re.search(r"Informe N\.º <b>MP-\d{8}-[0-9A-F]{6}</b>", html)
    assert html.count("class='fila-roi") >= res.n_rois  # una fila por partícula


def test_informe_sin_verdad_no_tiene_verificacion(tmp_path):
    canales, _ = generar_imagen_muestra(semilla=3)
    df = generar_calibracion("fusion", n_por_polimero=60, semilla=0)
    cal = Calibracion.desde_dataframe(df, columnas=_columnas("fusion"))
    res = analizar_muestra(canales, cal, estrategia="centroide")
    html = generar_informe_html(res, canales, tmp_path / "x.html", nombre_muestra="ambiental",
                                exportar_pdf=False).read_text(encoding="utf-8")
    assert "Verificación con composición conocida" not in html
    assert "no informada" in html  # sin escala espacial


def test_alertas_marcan_phasor_invalido_y_umbral(resultado_con_verdad):
    res, _ = resultado_con_verdad
    feats = res.features.copy()
    feats.index = feats.index.astype(int)
    primero, segundo = feats.index[:2]
    feats.loc[primero, "score_rechazo"] = np.inf
    feats.loc[segundo, "score_rechazo"] = 0.85
    obs, _ = alertas(feats, res.labels)
    assert ("err", "phasor no válido") in obs[primero]
    assert ("warn", "cerca del umbral") in obs[segundo]


def test_ayuda_y_glosario():
    boton = ayuda("phasor")
    assert 'class="ayuda"' in boton and "Phasor" in boton
    assert all(len(t) == 2 and t[0] and t[1] for t in GLOSARIO.values())
    with pytest.raises(KeyError):
        ayuda("no_existe")


def test_exportar_pdf_respeta_la_variable_de_entorno(tmp_path, monkeypatch):
    monkeypatch.setenv("NAPARI_MP_SIN_PDF", "1")
    ruta = tmp_path / "a.html"
    ruta.write_text("<html></html>")
    assert exportar_a_pdf(ruta) is None
