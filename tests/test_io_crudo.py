"""Tests de :mod:`napari_mp_classifier.io_crudo`.

No dependen de archivos ``.sdt`` / ``.czi`` reales: se sustituyen los lectores de
``phasorpy`` por señales sintéticas con la misma interfaz (``.values`` / ``.dims`` /
``.attrs`` / ``.coords``).
"""

from __future__ import annotations

import numpy as np
import pytest

from napari_mp_classifier.io_crudo import (
    PhasoresImagen,
    _reducir_a_2d,
    phasores_desde_czi,
    phasores_desde_sdt,
)


class _Coord:
    def __init__(self, values):
        self.values = np.asarray(values)


class _SenalFalsa:
    """Imita el objeto que devuelven ``signal_from_sdt`` / ``signal_from_czi``."""

    def __init__(self, values, dims, attrs=None, coords=None):
        self.values = np.asarray(values)
        self.dims = tuple(dims)
        self.shape = self.values.shape
        self.attrs = attrs or {}
        self.coords = {k: _Coord(v) for k, v in (coords or {}).items()}


def _cubo_flim(alto=6, ancho=5, bins=32, tau_bins=4.0):
    """Decaimiento monoexponencial idéntico en cada píxel, eje ``(Y, X, H)``."""
    t = np.arange(bins)
    decaimiento = np.exp(-t / tau_bins)
    return np.broadcast_to(decaimiento, (alto, ancho, bins)).copy()


def _cubo_espectral(alto=6, ancho=5, canales=16):
    """Gaussiana espectral idéntica en cada píxel, eje ``(C, Y, X)``."""
    c = np.arange(canales)
    espectro = np.exp(-((c - canales / 2) ** 2) / (2 * 3.0**2))
    return np.broadcast_to(espectro[:, None, None], (canales, alto, ancho)).copy()


# --------------------------------------------------------------------- helpers puros


def test_reducir_a_2d_ordena_a_yx_eje():
    datos = np.zeros((3, 6, 5))  # (H, Y, X)
    cubo, eje = _reducir_a_2d(datos, ("H", "Y", "X"), "H")
    assert cubo.shape == (6, 5, 3)
    assert eje == 2


def test_reducir_a_2d_suma_ejes_sobrantes():
    datos = np.ones((2, 6, 5, 3))  # (Q, Y, X, H): el eje Q se integra
    cubo, _eje = _reducir_a_2d(datos, ("Q", "Y", "X", "H"), "H")
    assert cubo.shape == (6, 5, 3)
    assert np.allclose(cubo, 2.0)


# ----------------------------------------------------------------------------- FLIM


def test_phasores_desde_sdt_basico(monkeypatch):
    senal = _SenalFalsa(_cubo_flim(), ("Y", "X", "H"), attrs={"frequency": 60.0})
    import phasorpy.io as _pio

    monkeypatch.setattr(_pio, "signal_from_sdt", lambda ruta: senal, raising=False)

    ph = phasores_desde_sdt("no_existe.sdt")
    assert isinstance(ph, PhasoresImagen)
    assert ph.modalidad == "flim"
    assert ph.calibrado is False
    assert ph.frecuencia_mhz == 60.0
    assert ph.forma == (6, 5)
    g, s, intensidad = ph
    assert g.shape == s.shape == intensidad.shape == (6, 5)
    # Señal homogénea -> phasor constante y sobre el semicírculo universal.
    assert np.allclose(g, g.flat[0])
    assert np.all(intensidad > 0)


def test_phasores_desde_sdt_sin_frecuencia(monkeypatch):
    senal = _SenalFalsa(_cubo_flim(), ("Y", "X", "H"), attrs={})
    import phasorpy.io as _pio

    monkeypatch.setattr(_pio, "signal_from_sdt", lambda ruta: senal, raising=False)
    with pytest.raises(ValueError, match="frecuencia"):
        phasores_desde_sdt("x.sdt")
    # Pero se puede pasar a mano.
    ph = phasores_desde_sdt("x.sdt", frecuencia_mhz=80.0)
    assert ph.frecuencia_mhz == 80.0


def test_phasores_desde_sdt_corregir_desfase(monkeypatch):
    # Decaimiento con el pico corrido 5 bins: sin corregir el phasor rota; al corregir
    # vuelve a caer sobre el semicírculo universal (|(g-0.5, s)| == 0.5).
    cubo = np.roll(_cubo_flim(bins=64, tau_bins=6.0), 5, axis=2)
    senal = _SenalFalsa(cubo, ("Y", "X", "H"), attrs={"frequency": 60.0})
    import phasorpy.io as _pio

    monkeypatch.setattr(_pio, "signal_from_sdt", lambda ruta: senal, raising=False)

    crudo = phasores_desde_sdt("x.sdt")
    corr = phasores_desde_sdt("x.sdt", corregir_desfase=True)
    assert crudo.metadatos["desfase_corregido"] is False
    assert corr.metadatos["desfase_corregido"] is True
    assert corr.calibrado is False

    def _dist_semicirculo(ph):
        return abs(np.hypot(ph.g.flat[0] - 0.5, ph.s.flat[0]) - 0.5)

    # Un decaimiento discreto no cae exacto sobre el semicírculo, pero corregir el
    # desfase lo acerca mucho respecto del crudo (pico corrido).
    assert _dist_semicirculo(corr) < 0.03
    assert _dist_semicirculo(corr) < _dist_semicirculo(crudo) / 3


def test_phasores_desde_sdt_calibra_con_referencia(monkeypatch):
    muestra = _SenalFalsa(_cubo_flim(tau_bins=4.0), ("Y", "X", "H"),
                          attrs={"frequency": 60.0})
    referencia = _SenalFalsa(_cubo_flim(tau_bins=2.0), ("Y", "X", "H"),
                             attrs={"frequency": 60.0})

    def _fake(ruta):
        return referencia if "ref" in str(ruta) else muestra

    import phasorpy.io as _pio

    monkeypatch.setattr(_pio, "signal_from_sdt", _fake, raising=False)
    ph = phasores_desde_sdt("m.sdt", ruta_referencia="ref.sdt",
                            referencia_lifetime_ns=2.0)
    assert ph.calibrado is True
    assert ph.forma == (6, 5)


# ------------------------------------------------------------------------- espectral


def test_phasores_desde_czi_basico(monkeypatch):
    senal = _SenalFalsa(
        _cubo_espectral(), ("C", "Y", "X"),
        coords={"C": np.linspace(450, 700, 16)},
    )
    import phasorpy.io as _pio

    monkeypatch.setattr(_pio, "signal_from_czi", lambda ruta: senal, raising=False)
    ph = phasores_desde_czi("x.czi")
    assert ph.modalidad == "espectral"
    assert ph.calibrado is True
    assert ph.frecuencia_mhz is None
    assert ph.metadatos["n_canales"] == 16
    assert ph.metadatos["rango_nm"] == (450.0, 700.0)
    assert ph.forma == (6, 5)


def test_phasores_desde_czi_rechaza_pocos_canales(monkeypatch):
    senal = _SenalFalsa(np.ones((2, 6, 5)), ("C", "Y", "X"),
                        coords={"C": [500, 600]})
    import phasorpy.io as _pio

    monkeypatch.setattr(_pio, "signal_from_czi", lambda ruta: senal, raising=False)
    with pytest.raises(ValueError, match="canales"):
        phasores_desde_czi("x.czi")


def test_phasores_imagen_desempaque():
    g = np.zeros((2, 2))
    ph = PhasoresImagen(g=g, s=g + 1, intensidad=g + 2)
    a, b, c = ph
    assert a[0, 0] == 0 and b[0, 0] == 1 and c[0, 0] == 2


# --------------------------------------------------------- varios detectores + IRF


def _sdt_con_bloques(monkeypatch, bloques):
    import phasorpy.io as _pio

    def _fake(ruta, index=0):
        if index >= len(bloques):
            raise IndexError(index)
        return bloques[index]

    monkeypatch.setattr(_pio, "signal_from_sdt", _fake, raising=False)


def test_sdt_elige_el_detector_con_mas_fotones(monkeypatch):
    debil = _SenalFalsa(_cubo_flim() * 1, ("Y", "X", "H"), attrs={"frequency": 60.0})
    fuerte = _SenalFalsa(_cubo_flim() * 20, ("Y", "X", "H"), attrs={"frequency": 60.0})
    _sdt_con_bloques(monkeypatch, [debil, fuerte])

    from napari_mp_classifier.io_crudo import listar_detectores_sdt

    detectores = listar_detectores_sdt("x.sdt")
    assert [d["indice"] for d in detectores] == [0, 1]
    assert detectores[1]["fotones"] > detectores[0]["fotones"]

    ph = phasores_desde_sdt("x.sdt")
    assert ph.metadatos["detector"] == 1
    assert len(ph.metadatos["fotones_por_detector"]) == 2
    assert phasores_desde_sdt("x.sdt", detector=0).metadatos["detector"] == 0
    with pytest.raises(ValueError, match="detector"):
        phasores_desde_sdt("x.sdt", detector=5)


def test_sdt_calibrar_irf(monkeypatch):
    from napari_mp_classifier.calibracion_flim import simular_decaimiento

    decaimiento, irf = simular_decaimiento(3.0, 59.96, n_bins=256, t0_ns=2.0, fotones=1e6,
                                           ventana=(0, 255), semilla=0)
    cubo = np.broadcast_to(decaimiento, (6, 5, 256)).copy()
    _sdt_con_bloques(monkeypatch, [_SenalFalsa(cubo, ("Y", "X", "H"), attrs={"frequency": 59.96})])

    ph = phasores_desde_sdt("x.sdt", calibrar_irf=True)
    assert ph.calibrado is True
    assert ph.metadatos["calibracion"] == "irf_flanco"
    assert abs(np.hypot(ph.g.flat[0] - 0.5, ph.s.flat[0]) - 0.5) < 0.05

    medida = phasores_desde_sdt("x.sdt", irf=irf)
    assert medida.metadatos["calibracion"] == "irf_medida"


def test_fotones_es_intensidad_por_canales(monkeypatch):
    senal = _SenalFalsa(_cubo_flim(bins=32), ("Y", "X", "H"), attrs={"frequency": 60.0})
    import phasorpy.io as _pio

    monkeypatch.setattr(_pio, "signal_from_sdt", lambda ruta: senal, raising=False)
    ph = phasores_desde_sdt("x.sdt")
    assert np.allclose(ph.fotones, senal.values.sum(axis=-1))
