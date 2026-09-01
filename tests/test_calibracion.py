"""Tests de :mod:`napari_mp_classifier.calibracion`."""

import numpy as np
import pandas as pd
import pytest

from napari_mp_classifier.calibracion import Calibracion
from datos_sinteticos import generar_calibracion


def test_desde_dataframe_flim():
    df = generar_calibracion("flim", n_por_polimero=50, semilla=0)
    cal = Calibracion.desde_dataframe(df, columnas=["g_flim", "s_flim"], columna_etiqueta="polimero")

    assert set(cal.etiquetas) == {"PET", "HDPE", "PVC", "LDPE", "PP", "PS"}
    assert cal.n_features == 2
    assert cal.matriz_centroides().shape == (6, 2)
    for etiqueta in cal.etiquetas:
        assert cal.covarianzas[etiqueta].shape == (2, 2)
        assert cal.n_muestras[etiqueta] == 50


def test_fusion_tiene_4_features():
    df = generar_calibracion("fusion", n_por_polimero=30)
    cal = Calibracion.desde_dataframe(
        df, columnas=["g_flim", "s_flim", "g_esp", "s_esp"], columna_etiqueta="polimero"
    )
    assert cal.n_features == 4
    assert cal.matriz_centroides().shape == (6, 4)


def test_columna_faltante_falla():
    df = pd.DataFrame({"g_flim": [0.1, 0.2], "polimero": ["PET", "PET"]})
    with pytest.raises(ValueError, match="columnas"):
        Calibracion.desde_dataframe(df, columnas=["g_flim", "s_flim"])


def test_pocas_muestras_falla():
    df = pd.DataFrame({"g_flim": [0.1], "s_flim": [0.2], "polimero": ["PET"]})
    with pytest.raises(ValueError, match="al menos 2"):
        Calibracion.desde_dataframe(df, columnas=["g_flim", "s_flim"])


def test_ida_y_vuelta_csv(tmp_path):
    df = generar_calibracion("flim", n_por_polimero=20)
    cal = Calibracion.desde_dataframe(df, columnas=["g_flim", "s_flim"])
    ruta = tmp_path / "cal.csv"
    cal.guardar_csv(ruta)
    recargado = pd.read_csv(ruta)
    assert set(recargado["polimero"]) == set(cal.etiquetas)
    assert {"g_flim", "s_flim", "n_muestras"}.issubset(recargado.columns)


def test_centroides_cercanos_a_los_teoricos():
    from datos_sinteticos import centroides_referencia

    df = generar_calibracion("flim", n_por_polimero=200, sigma=0.01, semilla=3)
    cal = Calibracion.desde_dataframe(df, columnas=["g_flim", "s_flim"])
    teoricos = centroides_referencia("flim")
    for etiqueta, centro in cal.centroides.items():
        assert np.allclose(centro, teoricos[etiqueta], atol=0.01)
