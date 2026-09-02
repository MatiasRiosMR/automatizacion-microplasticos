"""Tests de robustez (Fase 5): envejecimiento, ruido y flujo de fagocitos.

Todo sobre datos sintéticos. La calibración representa el estándar de referencia
(polímero tratado con abrasión + H2O2 [+ UV]); ``grado_envejecimiento`` mide el desajuste
del estado de la muestra respecto de ese estándar.
"""

import numpy as np
import pytest
from datos_sinteticos import (
    _columnas,
    centroides_referencia,
    generar_calibracion,
    generar_imagen_muestra,
    generar_mascara_celular,
    generar_particulas,
)

from napari_mp_classifier import Calibracion, ClasificadorPhasor
from napari_mp_classifier.segmentacion import restringir_a_mascara, segmentar


def _clasificador(modalidad="fusion", confianza=0.99):
    df = generar_calibracion(modalidad, n_por_polimero=80, sigma=0.02, semilla=0)
    cal = Calibracion.desde_dataframe(df, columnas=_columnas(modalidad))
    clf = ClasificadorPhasor(cal, estrategia="knn", confianza=confianza)
    clf.entrenar(df[_columnas(modalidad)].to_numpy(), df["polimero"].to_numpy())
    return clf


def test_grado_cero_no_cambia_nada():
    X0, y0 = generar_particulas("fusion", grado_envejecimiento=0.0, semilla=3)
    X1, y1 = generar_particulas("fusion", semilla=3)
    assert np.allclose(X0, X1)
    assert np.array_equal(y0, y1)


def test_envejecimiento_acerca_los_clusters_al_centro_comun():
    centros = centroides_referencia("fusion")
    centro_global = np.mean(list(centros.values()), axis=0)

    def dispersion(grado):
        X, y = generar_particulas("fusion", n_por_polimero=200, n_no_clasificables=0,
                                  grado_envejecimiento=grado, sigma=0.001, semilla=1)
        centroides = np.array([X[y == p].mean(axis=0) for p in centros])
        return np.mean(np.linalg.norm(centroides - centro_global, axis=1))

    # Más envejecimiento -> clusters más juntos (convergen a la firma común).
    assert dispersion(0.3) < dispersion(0.1) < dispersion(0.0)


@pytest.mark.parametrize("grado", [0.0, 0.1, 0.2])
def test_degradacion_monotona_y_modo_de_falla_conservador(grado):
    clf = _clasificador()
    X, y = generar_particulas("fusion", n_por_polimero=80, n_no_clasificables=80,
                              grado_envejecimiento=grado, semilla=5)
    pred = clf.predecir(X)
    pol = y != "no_clasificable"
    exactitud = (pred[pol] == y[pol]).mean()

    if grado == 0.0:
        assert exactitud > 0.97
    # Cuando falla, manda a "no_clasificable" antes que a otro polímero:
    mal = pred[pol] != y[pol]
    a_no_clasificable = pred[pol][mal] == "no_clasificable"
    if mal.any():
        assert a_no_clasificable.mean() > 0.5
    # La materia orgánica se sigue rechazando bien, no la afecta la deriva.
    assert (pred[~pol] == "no_clasificable").mean() > 0.9


def test_degradacion_es_monotona_con_el_grado():
    clf = _clasificador()
    exactitudes = []
    for grado in (0.0, 0.15, 0.3):
        X, y = generar_particulas("fusion", n_por_polimero=100, n_no_clasificables=0,
                                  grado_envejecimiento=grado, semilla=4)
        exactitudes.append((clf.predecir(X) == y).mean())
    assert exactitudes[0] >= exactitudes[1] >= exactitudes[2]


def test_fusion_mas_robusta_que_una_sola_modalidad_bajo_envejecimiento():
    clfs = {m: _clasificador(m, confianza=None) for m in ("flim", "espectral", "fusion")}
    for grado in (0.1, 0.2):
        exac = {}
        for m in ("flim", "espectral", "fusion"):
            X, y = generar_particulas(m, n_por_polimero=100, n_no_clasificables=0,
                                      grado_envejecimiento=grado, semilla=8)
            exac[m] = (clfs[m].predecir(X) == y).mean()
        assert exac["fusion"] >= max(exac["flim"], exac["espectral"]) - 0.02


def test_confianza_alta_recupera_polimero_bajo_desajuste():
    X, y = generar_particulas("fusion", n_por_polimero=100, n_no_clasificables=100,
                              grado_envejecimiento=0.12, semilla=7)
    pol = y != "no_clasificable"

    perdidos = {}
    for confianza in (0.95, 0.999):
        pred = _clasificador(confianza=confianza).predecir(X)
        perdidos[confianza] = (pred[pol] == "no_clasificable").mean()
    assert perdidos[0.999] < perdidos[0.95]


def test_ruido_alto_degrada_pero_no_misclasifica():
    clf = _clasificador()
    X, y = generar_particulas("fusion", n_por_polimero=80, n_no_clasificables=0,
                              sigma=0.08, semilla=6)
    pred = clf.predecir(X)
    mal = pred != y
    # Bajo mucho ruido se pierde exactitud, pero sobre todo hacia "no_clasificable".
    if mal.any():
        assert (pred[mal] == "no_clasificable").mean() > 0.6


def test_flujo_fagocitos_restringe_a_las_celulas():
    canales, verdad = generar_imagen_muestra(semilla=21)
    mascara = generar_mascara_celular(canales["intensidad"].shape, verdad,
                                      fraccion_fagocitada=0.6, semilla=21)
    labels = segmentar(canales["intensidad"], g_flim=canales["g_flim"], s_flim=canales["s_flim"])
    dentro = restringir_a_mascara(labels, mascara)

    assert 0 < dentro.max() < labels.max()
    # Todas las ROIs conservadas están efectivamente dentro de la máscara.
    for etiqueta in np.unique(dentro[dentro > 0]):
        roi = dentro == etiqueta
        assert (roi & mascara).sum() / roi.sum() >= 0.5


def test_mascara_celular_vacia_si_no_hay_fagocitosis():
    canales, verdad = generar_imagen_muestra(semilla=2)
    mascara = generar_mascara_celular(canales["intensidad"].shape, verdad,
                                      fraccion_fagocitada=0.0, semilla=2)
    assert not mascara.any()
