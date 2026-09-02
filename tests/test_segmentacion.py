"""Tests de :mod:`napari_mp_classifier.segmentacion`."""

import numpy as np
import pytest
from datos_sinteticos import generar_imagen_muestra

from napari_mp_classifier.metricas import evaluar_segmentacion
from napari_mp_classifier.segmentacion import (
    restringir_a_mascara,
    segmentar,
    segmentar_kmeans,
    segmentar_umbral,
    separar_contacto,
)


@pytest.fixture
def muestra():
    return generar_imagen_muestra(semilla=0)


def test_umbral_detecta_particulas(muestra):
    canales, verdad = muestra
    labels = segmentar_umbral(canales["intensidad"])
    assert labels.max() > 10
    assert labels.min() == 0
    # Sin watershed, el umbralado crudo ya empareja la mayoría de las ROIs con buen IoU;
    # el recall sube al separar las partículas en contacto (ver test_segmentar_recall_iou).
    rep = evaluar_segmentacion(labels, verdad["labels"], iou_min=0.5)
    assert rep.iou_medio > 0.65


@pytest.mark.parametrize("semilla", [0, 1, 2])
def test_segmentar_recall_iou(semilla):
    canales, verdad = generar_imagen_muestra(semilla=semilla)
    labels = segmentar(
        canales["intensidad"],
        g_flim=canales["g_flim"],
        s_flim=canales["s_flim"],
        metodo="umbral",
    )
    rep = evaluar_segmentacion(labels, verdad["labels"], iou_min=0.5)
    # IoU comparable a FIMAP (Ho et al. 2025: 0,877).
    assert rep.iou_medio > 0.7
    assert rep.recall_deteccion > 0.6
    assert rep.precision_deteccion > 0.7


def test_kmeans_usa_phasor(muestra):
    canales, verdad = muestra
    labels = segmentar(
        canales["intensidad"],
        g_flim=canales["g_flim"],
        s_flim=canales["s_flim"],
        metodo="kmeans",
    )
    rep = evaluar_segmentacion(labels, verdad["labels"], iou_min=0.5)
    assert rep.n_predichas > 5
    assert rep.iou_medio > 0.7
    # K-means se apoya en la firma de phasor: menos recall que el umbralado (descarta
    # cuerpos difusos de materia orgánica) pero precisión comparable.
    solo_intensidad = segmentar_kmeans(canales["intensidad"])
    assert solo_intensidad.max() > 5


def test_segmentar_metodo_invalido(muestra):
    canales, _ = muestra
    with pytest.raises(ValueError, match="metodo"):
        segmentar(canales["intensidad"], metodo="magia")


@pytest.mark.parametrize("metodo", ["umbral", "kmeans"])
def test_segmentar_end_to_end(metodo, muestra):
    canales, _ = muestra
    labels = segmentar(
        canales["intensidad"],
        g_flim=canales["g_flim"],
        s_flim=canales["s_flim"],
        g_esp=canales["g_esp"],
        s_esp=canales["s_esp"],
        metodo=metodo,
    )
    assert labels.shape == canales["intensidad"].shape
    # Reindexado contiguo 1..n.
    ids = np.unique(labels[labels > 0])
    assert list(ids) == list(range(1, len(ids) + 1))


def test_filtro_de_tamano():
    img = np.zeros((40, 40))
    img[5:8, 5:8] = 1.0        # 9 px  -> se descarta con tam_min=20
    img[15:28, 15:28] = 1.0    # 169 px -> se conserva
    labels = segmentar(img, metodo="umbral", separar=False, tam_min=20, cerrar=0)
    assert labels.max() == 1


def test_tam_max_descarta_cuerpos_grandes():
    img = np.zeros((60, 60))
    img[5:12, 5:12] = 1.0       # 49 px
    img[20:50, 20:50] = 1.0     # 900 px
    labels = segmentar(img, metodo="umbral", separar=False, tam_min=10, tam_max=200, cerrar=0)
    areas = np.bincount(labels.ravel())[1:]
    assert all(a <= 200 for a in areas)
    assert labels.max() == 1


def test_separar_contacto_parte_dos_particulas():
    # Dos discos que se tocan -> una sola componente conexa que watershed debe partir.
    img = np.zeros((60, 100), dtype=bool)
    yy, xx = np.mgrid[0:60, 0:100]
    img |= (yy - 30) ** 2 + (xx - 38) ** 2 < 15**2
    img |= (yy - 30) ** 2 + (xx - 60) ** 2 < 15**2
    from skimage.measure import label

    assert label(img).max() == 1
    separadas = separar_contacto(img, img.astype(float))
    assert separadas.max() == 2


def test_restringir_a_mascara_celular():
    labels = np.zeros((50, 50), dtype=int)
    labels[5:15, 5:15] = 1     # dentro de la célula
    labels[30:40, 30:40] = 2   # fuera
    mascara = np.zeros((50, 50), dtype=bool)
    mascara[0:20, 0:20] = True

    filtrado = restringir_a_mascara(labels, mascara, solape_min=0.5)
    assert set(np.unique(filtrado)) == {0, 1}


def test_reproducible(muestra):
    canales_a, verdad_a = muestra
    canales_b, verdad_b = generar_imagen_muestra(semilla=0)
    assert np.allclose(canales_a["intensidad"], canales_b["intensidad"])
    assert np.array_equal(verdad_a["labels"], verdad_b["labels"])
