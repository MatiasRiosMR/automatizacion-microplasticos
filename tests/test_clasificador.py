"""Tests de :mod:`napari_mp_classifier.clasificador`."""

import numpy as np
import pytest
from datos_sinteticos import generar_calibracion, generar_particulas

from napari_mp_classifier import NO_CLASIFICABLE
from napari_mp_classifier.calibracion import Calibracion
from napari_mp_classifier.clasificador import ClasificadorPhasor


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


# ------------------------------------------------------------- correcciones de la auditoría


@pytest.mark.parametrize("estrategia", ["centroide", "knn", "gmm"])
def test_phasor_nan_es_no_clasificable(estrategia):
    # Antes: 'centroide' asignaba el primer polímero en silencio y knn/gmm fallaban.
    cal, df = _calibracion("flim", n_por_polimero=40)
    clf = ClasificadorPhasor(cal, estrategia=estrategia)
    clf.entrenar(df[["g_flim", "s_flim"]].to_numpy(), df["polimero"].to_numpy())
    X = np.array([[np.nan, np.nan], cal.centroides["PS"]])
    pred, score = clf.predecir_con_score(X)
    assert pred[0] == NO_CLASIFICABLE and np.isinf(score[0])
    assert pred[1] == "PS"


def test_umbral_hotelling_mantiene_la_tasa_nominal_con_pocas_mediciones():
    # Con n=10 mediciones en 4D, χ² rechaza ~20 % de polímero real; Hotelling ~1 %.
    from datos_sinteticos import _columnas

    tasas = {"hotelling": [], "chi2": []}
    for semilla in range(20):
        df = generar_calibracion("fusion", n_por_polimero=10, sigma=0.02, semilla=semilla)
        cal = Calibracion.desde_dataframe(df, columnas=_columnas("fusion"))
        X, _ = generar_particulas("fusion", n_por_polimero=100, n_no_clasificables=0,
                                  sigma=0.02, semilla=1000 + semilla)
        for umbral, lista in tasas.items():
            pred = ClasificadorPhasor(cal, umbral=umbral).predecir(X)
            lista.append((pred == NO_CLASIFICABLE).mean())
    assert np.mean(tasas["hotelling"]) < 0.03
    assert np.mean(tasas["chi2"]) > 0.10


def test_umbral_hotelling_converge_a_chi2():
    from scipy.stats import chi2

    from napari_mp_classifier.clasificador import umbral_hotelling

    assert umbral_hotelling(10, 4, 0.99) > umbral_hotelling(100, 4, 0.99)
    assert umbral_hotelling(100_000, 4, 0.99) == pytest.approx(chi2.ppf(0.99, 4), rel=1e-3)
    assert np.isinf(umbral_hotelling(4, 4, 0.99))


def test_pocas_mediciones_para_la_dimension_avisa():
    from datos_sinteticos import _columnas

    df = generar_calibracion("fusion", n_por_polimero=4, semilla=0)
    cal = Calibracion.desde_dataframe(df, columnas=_columnas("fusion"))
    with pytest.warns(UserWarning, match="mediciones"):
        ClasificadorPhasor(cal)


def test_qda_no_favorece_al_cluster_mas_ancho():
    # Dos clusters 1D-equivalentes: A compacto, B muy disperso. Un punto a 2,5σ de A
    # pertenece a A; por distancia de Mahalanobis pura se lo lleva B.
    cal = Calibracion(
        centroides={"A": np.array([0.0, 0.0]), "B": np.array([1.0, 0.0])},
        covarianzas={"A": np.eye(2) * 0.01, "B": np.eye(2) * 1.0},
        columnas=["g_flim", "s_flim"],
        n_muestras={"A": 500, "B": 500},
    )
    x = np.array([[0.25, 0.0]])
    assert ClasificadorPhasor(cal, confianza=None, regla="qda").predecir(x)[0] == "A"
    assert ClasificadorPhasor(cal, confianza=None, regla="mahalanobis").predecir(x)[0] == "B"


def test_voto_knn_desempata_por_cercania_no_por_orden_alfabetico():
    from napari_mp_classifier.clasificador import _voto_ponderado

    etiquetas = np.array(["PS", "PS", "HDPE", "HDPE", "PET"])
    distancias = np.array([0.01, 0.02, 0.03, 0.03, 0.05])
    assert _voto_ponderado(etiquetas, distancias) == "PS"


def test_knn_conformal_tiene_cobertura_cercana_a_la_confianza():
    cal, df = _calibracion("flim", n_por_polimero=150, sigma=0.02, semilla=0)
    clf = ClasificadorPhasor(cal, estrategia="knn", umbral_knn="conformal", confianza=0.95)
    clf.entrenar(df[["g_flim", "s_flim"]].to_numpy(), df["polimero"].to_numpy())
    X, _ = generar_particulas("flim", n_por_polimero=150, n_no_clasificables=0, sigma=0.02,
                              semilla=3)
    rechazo = (clf.predecir(X) == NO_CLASIFICABLE).mean()
    assert rechazo < 0.12


def test_gmm_supervisado_un_componente_por_polimero():
    cal, df = _calibracion("fusion", n_por_polimero=60, sigma=0.04, semilla=0)
    from datos_sinteticos import _columnas

    clf = ClasificadorPhasor(cal, estrategia="gmm", confianza=None)
    clf.entrenar(df[_columnas("fusion")].to_numpy(), df["polimero"].to_numpy())
    X, y = generar_particulas("fusion", n_por_polimero=50, n_no_clasificables=0, sigma=0.04,
                              semilla=5)
    pred = clf.predecir(X)
    assert set(pred) == set(cal.etiquetas)  # ningún polímero queda sin componente
    assert (pred == y).mean() > 0.85


def test_gmm_sin_etiquetas_falla_con_mensaje_claro():
    cal, df = _calibracion("flim")
    with pytest.raises(ValueError, match="supervisada"):
        ClasificadorPhasor(cal, estrategia="gmm").entrenar(df[["g_flim", "s_flim"]].to_numpy())


def test_knn_sin_entrenar_falla():
    cal, _ = _calibracion("flim")
    with pytest.raises(RuntimeError, match="entrenar"):
        ClasificadorPhasor(cal, estrategia="knn").predecir(np.zeros((1, 2)))
