"""Tests de :mod:`napari_mp_classifier.clasificador`."""

import numpy as np
import pytest

from napari_mp_classifier import NO_CLASIFICABLE
from napari_mp_classifier.calibracion import Calibracion
from napari_mp_classifier.clasificador import ClasificadorPhasor
from datos_sinteticos import generar_calibracion, generar_particulas


def _calibracion(modalidad="flim", **kw):
    df = generar_calibracion(modalidad, **kw)
    from datos_sinteticos import _columnas

    return Calibracion.desde_dataframe(df, columnas=_columnas(modalidad)), df


@pytest.mark.parametrize("estrategia", ["centroide", "knn", "gmm"])
@pytest.mark.parametrize("modalidad", ["espectral", "fusion"])
def test_clasifica_polimeros_conocidos(estrategia, modalidad):
    cal, df = _calibracion(modalidad, n_por_polimero=80, semilla=0)
    from datos_sinteticos import _columnas

    clf = ClasificadorPhasor(cal, estrategia=estrategia, confianza=None)
    if estrategia == "knn":
        clf.entrenar(df[_columnas(modalidad)].to_numpy(), df["polimero"].to_numpy())
    else:
        clf.entrenar()

    X, y = generar_particulas(modalidad, n_por_polimero=50, n_no_clasificables=0, semilla=5)
    exactitud = (clf.predecir(X) == y).mean()
    assert exactitud > 0.9


def test_flim_solo_tiene_pares_solapados_que_la_fusion_resuelve():
    # FLIM-only: PVC/PET y LDPE/HDPE tienen lifetimes cercanos y se confunden.
    # La fusión con la modalidad espectral debe recuperar esa separación.
    exact = {}
    for modalidad in ("flim", "fusion"):
        cal, _ = _calibracion(modalidad, n_por_polimero=80, semilla=0)
        clf = ClasificadorPhasor(cal, confianza=None).entrenar()
        from datos_sinteticos import _columnas  # noqa: F401

        X, y = generar_particulas(modalidad, n_por_polimero=50, n_no_clasificables=0, semilla=5)
        exact[modalidad] = (clf.predecir(X) == y).mean()
    assert exact["fusion"] >= exact["flim"]


@pytest.mark.parametrize("estrategia", ["centroide", "knn", "gmm"])
def test_no_clasificable_rechaza_materia_organica(estrategia):
    cal, df = _calibracion("flim", n_por_polimero=80)
    clf = ClasificadorPhasor(cal, estrategia=estrategia, confianza=0.99)
    if estrategia == "knn":
        clf.entrenar(df[["g_flim", "s_flim"]].to_numpy(), df["polimero"].to_numpy())
    else:
        clf.entrenar()

    X, y = generar_particulas("flim", n_por_polimero=40, n_no_clasificables=80, semilla=7)
    pred = clf.predecir(X)

    es_organico = y == NO_CLASIFICABLE
    # La mayoría de la materia orgánica se rechaza...
    assert (pred[es_organico] == NO_CLASIFICABLE).mean() > 0.8
    # ...y casi ningún polímero real se pierde como "no clasificable".
    assert (pred[~es_organico] == NO_CLASIFICABLE).mean() < 0.1


def test_confianza_none_no_rechaza_nada():
    cal, _ = _calibracion("flim")
    clf = ClasificadorPhasor(cal, confianza=None).entrenar()
    X, _ = generar_particulas("flim", n_no_clasificables=50)
    assert NO_CLASIFICABLE not in set(clf.predecir(X))


def test_umbral_dimension_aware():
    # El umbral de Mahalanobis² debe crecer con la dimensión (chi² con más grados de libertad).
    cal2, _ = _calibracion("flim")
    cal4, _ = _calibracion("fusion")
    u2 = ClasificadorPhasor(cal2, confianza=0.99).umbral_mahalanobis2_
    u4 = ClasificadorPhasor(cal4, confianza=0.99).umbral_mahalanobis2_
    assert u4 > u2


def test_score_crece_con_la_distancia():
    cal, _ = _calibracion("flim")
    clf = ClasificadorPhasor(cal).entrenar()
    centro = cal.centroides["PS"]
    X = np.vstack([centro, centro + 0.5])
    _, score = clf.predecir_con_score(X)
    assert score[1] > score[0]


def test_forma_de_X_invalida_falla():
    cal, _ = _calibracion("flim")
    clf = ClasificadorPhasor(cal).entrenar()
    with pytest.raises(ValueError, match="forma"):
        clf.predecir(np.zeros((10, 3)))


def test_fusion_mejora_o_iguala_a_una_sola_modalidad():
    # La fusión FLIM+espectral no debería empeorar la separación de clusters.
    resultados = {}
    for modalidad in ("flim", "espectral", "fusion"):
        cal, _ = _calibracion(modalidad, n_por_polimero=80, semilla=0)
        clf = ClasificadorPhasor(cal, confianza=None).entrenar()
        X, y = generar_particulas(modalidad, n_por_polimero=60, n_no_clasificables=0, semilla=9)
        resultados[modalidad] = (clf.predecir(X) == y).mean()
    assert resultados["fusion"] >= max(resultados["flim"], resultados["espectral"]) - 0.02
