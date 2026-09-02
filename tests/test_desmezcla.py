"""Tests de :mod:`napari_mp_classifier.desmezcla`."""

import numpy as np
import pytest
from datos_sinteticos import (
    _columnas,
    centroides_referencia,
    generar_calibracion,
    generar_imagen_muestra,
)

from napari_mp_classifier import Calibracion
from napari_mp_classifier.desmezcla import (
    enmascarar_por_fraccion,
    fracciones_dos_componentes,
    fracciones_multi_componente,
    phasor_mp_de_calibracion,
)


@pytest.fixture
def componentes():
    cal = Calibracion.desde_dataframe(
        generar_calibracion("fusion", n_por_polimero=60), columnas=_columnas("fusion")
    )
    p_mp = phasor_mp_de_calibracion(cal, "esp")
    cf = np.mean(list(centroides_referencia("fusion").values()), axis=0) + 0.22
    p_auto = (cf[2], cf[3])
    return p_mp, p_auto


def test_fraccion_entre_0_y_1_y_nan_en_fondo(componentes):
    p_mp, p_auto = componentes
    canales, _ = generar_imagen_muestra(semilla=0)
    frac = fracciones_dos_componentes(canales["g_esp"], canales["s_esp"], p_mp, p_auto)

    assert frac.shape == canales["g_esp"].shape
    validos = np.isfinite(frac)
    assert (frac[validos] >= 0).all() and (frac[validos] <= 1).all()
    # El fondo (phasor NaN) queda NaN.
    assert np.isnan(frac[~np.isfinite(canales["g_esp"])]).all()


def test_particulas_de_polimero_tienen_mas_fraccion_mp_que_la_materia_organica(componentes):
    p_mp, p_auto = componentes
    canales, verdad = generar_imagen_muestra(semilla=2)
    frac = fracciones_dos_componentes(canales["g_esp"], canales["s_esp"], p_mp, p_auto)

    labels = verdad["labels"]
    es_polimero = np.zeros(labels.shape, dtype=bool)
    es_organico = np.zeros(labels.shape, dtype=bool)
    for etiqueta, codigo in verdad["polimero"].items():
        destino = es_organico if codigo == "no_clasificable" else es_polimero
        destino |= labels == etiqueta

    frac_pol = np.nanmean(frac[es_polimero])
    frac_org = np.nanmean(frac[es_organico])
    assert frac_pol > frac_org


def test_enmascarar_por_fraccion():
    frac = np.array([[0.1, 0.6], [np.nan, 0.9]])
    mascara = enmascarar_por_fraccion(frac, umbral=0.5)
    assert mascara.tolist() == [[False, True], [False, True]]


def test_fracciones_multi_componente_suma_aproximada(componentes):
    p_mp, p_auto = componentes
    canales, _ = generar_imagen_muestra(semilla=0)
    fr = fracciones_multi_componente(
        canales["intensidad"], canales["g_esp"], canales["s_esp"],
        {"MP": p_mp, "auto": p_auto},
    )
    assert set(fr) == {"MP", "auto"}
    total = np.nansum([fr["MP"], fr["auto"]], axis=0)
    validos = np.isfinite(fr["MP"]) & np.isfinite(fr["auto"])
    assert np.allclose(total[validos], 1.0, atol=0.15)


def test_phasor_mp_de_calibracion_modalidad_invalida():
    cal = Calibracion.desde_dataframe(
        generar_calibracion("flim", n_por_polimero=30), columnas=_columnas("flim")
    )
    with pytest.raises(ValueError, match="modalidad"):
        phasor_mp_de_calibracion(cal, "esp")
