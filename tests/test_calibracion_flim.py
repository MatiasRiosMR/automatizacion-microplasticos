"""Tests de :mod:`napari_mp_classifier.calibracion_flim` (calibración con IRF, estilo SPCImage).

Se validan sobre decaimientos simulados con la configuración de LAMAE (59,96 MHz, 1024
canales, ventana del TAC 5 %–96 %) y tiempos de vida conocidos.
"""

import numpy as np
import pytest
from phasorpy.lifetime import phasor_to_apparent_lifetime

from napari_mp_classifier.calibracion_flim import (
    calibrar_con_irf,
    irf_desde_flanco,
    phasor_de_histograma,
    simular_decaimiento,
)

FREC = 59.96


def _tau_fase(g, s):
    return float(phasor_to_apparent_lifetime(g, s, FREC)[0])


@pytest.mark.parametrize("tau", [1.0, 2.0, 3.0, 4.0])
def test_irf_del_flanco_recupera_el_lifetime(tau):
    decaimiento, _ = simular_decaimiento(tau, FREC, fotones=2e6, semilla=1)
    _, g, s = phasor_de_histograma(decaimiento)
    gc, sc, _ = calibrar_con_irf(g, s, irf_desde_flanco(decaimiento), FREC)
    assert _tau_fase(gc, sc) == pytest.approx(tau, rel=0.08)


@pytest.mark.parametrize("tau", [2.0, 3.0])
def test_corregir_pico_subestima_el_lifetime(tau):
    # El método «pico → 0» (anterior) ignora que el máximo llega después del pulso.
    decaimiento, _ = simular_decaimiento(tau, FREC, fotones=2e6, semilla=2)
    _, g, s = phasor_de_histograma(np.roll(decaimiento, -int(decaimiento.argmax())))
    assert _tau_fase(g, s) < 0.9 * tau


def test_irf_verdadera_lleva_al_semicirculo():
    decaimiento, irf = simular_decaimiento(2.5, FREC, fotones=5e6, ventana=(0, 1023), semilla=3)
    _, g, s = phasor_de_histograma(decaimiento)
    gc, sc, info = calibrar_con_irf(g, s, irf, FREC)
    assert abs(np.hypot(gc - 0.5, sc) - 0.5) < 0.01
    assert 0 < info["modulacion_irf"] <= 1


def test_irf_normalizada_y_antes_del_pico():
    decaimiento, _ = simular_decaimiento(3.0, FREC, fotones=1e6, semilla=4)
    irf = irf_desde_flanco(decaimiento)
    assert irf.sum() == pytest.approx(1.0)
    assert (irf >= 0).all()
    assert np.flatnonzero(irf).max() <= int(np.argmax(decaimiento)) + 2


def test_irf_sin_flanco_falla():
    with pytest.raises(ValueError, match="flanco"):
        irf_desde_flanco(np.zeros(256))


def test_calibrar_por_pixel_conserva_la_forma():
    g = np.full((4, 3), 0.1)
    s = np.full((4, 3), 0.6)
    decaimiento, _ = simular_decaimiento(2.0, FREC, fotones=1e6, semilla=5)
    gc, sc, _ = calibrar_con_irf(g, s, irf_desde_flanco(decaimiento), FREC)
    assert gc.shape == sc.shape == (4, 3)
