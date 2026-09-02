"""Segmentación: detección de ROIs candidatas a microplástico en imágenes de Nile Red.

Convierte una imagen de intensidad de Nile Red (con o sin coordenadas de phasor por
píxel) en una imagen de ``labels`` (enteros, ``0`` = fondo) compatible con napari, donde
cada label es una partícula candidata a microplástico.

Enfoques disponibles
--------------------
- :func:`segmentar_umbral` — umbralado global (Otsu / Li / valor fijo) + limpieza
  morfológica. Rápido y suficiente cuando el contraste señal/fondo es alto.
- :func:`segmentar_kmeans` — K-means por píxel sobre ``[intensidad, g, s]`` para separar
  **señal / fondo / sombra**, siguiendo FIMAP (Ho et al. 2025, IoU 87,7 %). Es más robusto
  a la materia orgánica difusa porque usa también la firma de phasor, no solo el brillo.
- :func:`separar_contacto` — ``watershed`` sobre la transformada de distancia para partir
  partículas que se tocan.
- :func:`segmentar` — orquesta lo anterior (umbral o K-means → limpieza → watershed →
  filtro de tamaño) y es el punto de entrada habitual.
- :func:`restringir_a_mascara` — se queda solo con las ROIs contenidas en una máscara
  celular, para las muestras de fagocitos (monocitos / neutrófilos): así se aísla la señal
  de NR-MP fagocitado de la autofluorescencia del resto del campo (Park et al. 2020).

La evaluación de la segmentación (IoU, precisión/recall de detección) vive en
:mod:`napari_mp_classifier.metricas`.
"""

from __future__ import annotations

import numpy as np

METODOS = ("umbral", "kmeans")


def _normalizar(imagen: np.ndarray) -> np.ndarray:
    """Reescala una imagen a ``[0, 1]`` (percentil 1–99, robusto a outliers)."""
    imagen = np.asarray(imagen, dtype=float)
    lo, hi = np.nanpercentile(imagen, [1.0, 99.0])
    if hi <= lo:
        return np.zeros_like(imagen)
    return np.clip((imagen - lo) / (hi - lo), 0.0, 1.0)


def _limpiar_mascara(mascara: np.ndarray, tam_min: int, cerrar: int) -> np.ndarray:
    """Cierra huecos chicos y descarta objetos menores a ``tam_min`` px."""
    from scipy import ndimage as ndi
    from skimage.measure import label
    from skimage.morphology import closing, disk

    mascara = np.asarray(mascara, dtype=bool)
    if cerrar > 0:
        mascara = closing(mascara, disk(cerrar))
    mascara = ndi.binary_fill_holes(mascara)

    etiquetas = label(mascara)
    if etiquetas.max() > 0:
        cuentas = np.bincount(etiquetas.ravel())
        chicos = {i for i in np.flatnonzero(cuentas < max(tam_min, 1)) if i != 0}
        if chicos:
            mascara = mascara & ~np.isin(etiquetas, list(chicos))
    return mascara


def segmentar_umbral(
    intensidad: np.ndarray,
    *,
    metodo: str = "otsu",
    factor: float = 1.0,
    tam_min: int = 8,
    cerrar: int = 1,
) -> np.ndarray:
    """Segmenta por umbralado global de la intensidad.

    Parameters
    ----------
    intensidad : numpy.ndarray, shape (alto, ancho)
        Imagen de intensidad de Nile Red.
    metodo : {"otsu", "li", "media"} or float, optional
        Regla del umbral. Un ``float`` se usa como umbral absoluto (sobre la intensidad
        normalizada a ``[0, 1]``).
    factor : float, optional
        Multiplica el umbral calculado (``>1`` es más restrictivo). Por defecto ``1.0``.
    tam_min : int, optional
        Área mínima en px para conservar un objeto. Por defecto ``8``.
    cerrar : int, optional
        Radio del cierre morfológico (``0`` lo desactiva). Por defecto ``1``.

    Returns
    -------
    numpy.ndarray of int, shape (alto, ancho)
        Imagen de labels (``0`` = fondo).
    """
    from skimage.filters import threshold_li, threshold_otsu
    from skimage.measure import label

    norm = _normalizar(intensidad)
    if isinstance(metodo, (int, float)) and not isinstance(metodo, bool):
        umbral = float(metodo)
    elif metodo == "otsu":
        umbral = float(threshold_otsu(norm))
    elif metodo == "li":
        umbral = float(threshold_li(norm))
    elif metodo == "media":
        umbral = float(norm.mean())
    else:
        raise ValueError(f"metodo debe ser 'otsu', 'li', 'media' o un número; recibí {metodo!r}")

    mascara = _limpiar_mascara(norm > umbral * factor, tam_min, cerrar)
    return label(mascara).astype(np.int32)


def segmentar_kmeans(
    intensidad: np.ndarray,
    g: np.ndarray | None = None,
    s: np.ndarray | None = None,
    *,
    n_clusters: int = 3,
    tam_min: int = 8,
    cerrar: int = 1,
    semilla: int = 0,
) -> np.ndarray:
    """Segmenta con K-means por píxel sobre ``[intensidad, g, s]`` (enfoque FIMAP).

    Agrupa los píxeles en ``n_clusters`` clases (típicamente señal / fondo / sombra) y se
    queda con el cluster de mayor intensidad media como "señal". Incluir las coordenadas
    de phasar ``g``/``s`` ayuda a separar partículas tenues de materia orgánica difusa que
    comparten brillo pero no firma.

    Parameters
    ----------
    intensidad : numpy.ndarray, shape (alto, ancho)
    g, s : numpy.ndarray, shape (alto, ancho), optional
        Coordenadas de phasor por píxel (de una modalidad; FLIM o espectral). Si se pasan,
        se suman como features. Los ``NaN`` (fondo) se rellenan con la mediana.
    n_clusters : int, optional
        Número de clusters de K-means. Por defecto ``3`` (señal / fondo / sombra).
    tam_min, cerrar : int, optional
        Ver :func:`segmentar_umbral`.
    semilla : int, optional

    Returns
    -------
    numpy.ndarray of int, shape (alto, ancho)
        Imagen de labels (``0`` = fondo).
    """
    from skimage.measure import label
    from sklearn.cluster import KMeans

    norm = _normalizar(intensidad)
    features = [norm.ravel()]
    for canal in (g, s):
        if canal is not None:
            col = np.asarray(canal, dtype=float).ravel()
            mediana = np.nanmedian(col) if np.isfinite(col).any() else 0.0
            features.append(np.nan_to_num(col, nan=mediana))
    X = np.column_stack(features)

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=semilla)
    etiquetas = km.fit_predict(X)
    intensidad_media = [norm.ravel()[etiquetas == c].mean() for c in range(n_clusters)]
    cluster_senal = int(np.argmax(intensidad_media))

    mascara = _limpiar_mascara((etiquetas == cluster_senal).reshape(norm.shape), tam_min, cerrar)
    return label(mascara).astype(np.int32)


def separar_contacto(
    mascara: np.ndarray,
    intensidad: np.ndarray | None = None,
    *,
    min_distancia: int = 3,
    umbral_marcador: float = 0.4,
) -> np.ndarray:
    """Separa partículas en contacto con ``watershed`` sobre la transformada de distancia.

    Parameters
    ----------
    mascara : numpy.ndarray
        Máscara binaria o de labels (se binariza) de la región de señal.
    intensidad : numpy.ndarray, optional
        Si se pasa, el relieve de inundación combina distancia e intensidad (los picos de
        brillo actúan como centros de partícula); si no, se usa solo la distancia.
    min_distancia : int, optional
        Separación mínima en px entre marcadores (máximos locales). Por defecto ``5``.
    umbral_marcador : float, optional
        Fracción del máximo de la transformada de distancia por debajo de la cual no se
        siembran marcadores. Por defecto ``0.35``.

    Returns
    -------
    numpy.ndarray of int
        Labels con las partículas separadas.
    """
    from scipy import ndimage as ndi
    from skimage.feature import peak_local_max
    from skimage.measure import label
    from skimage.segmentation import watershed

    binaria = np.asarray(mascara) > 0
    if not binaria.any():
        return np.zeros_like(binaria, dtype=np.int32)

    distancia = ndi.distance_transform_edt(binaria)
    coords = peak_local_max(
        distancia,
        min_distance=min_distancia,
        threshold_abs=umbral_marcador * distancia.max(),
        labels=binaria,
    )
    if len(coords) == 0:
        return label(binaria).astype(np.int32)

    semillas = np.zeros(binaria.shape, dtype=bool)
    semillas[tuple(coords.T)] = True
    marcadores = label(semillas)

    relieve = -distancia
    if intensidad is not None:
        relieve = relieve - _normalizar(intensidad)
    return watershed(relieve, marcadores, mask=binaria).astype(np.int32)


def segmentar(
    intensidad: np.ndarray,
    *,
    g_flim: np.ndarray | None = None,
    s_flim: np.ndarray | None = None,
    g_esp: np.ndarray | None = None,
    s_esp: np.ndarray | None = None,
    metodo: str = "umbral",
    separar: bool = True,
    tam_min: int = 8,
    tam_max: int | None = None,
    cerrar: int = 1,
    semilla: int = 0,
) -> np.ndarray:
    """Punto de entrada: imagen de intensidad (+ phasor) → labels de partículas.

    Encadena umbral o K-means → limpieza morfológica → ``watershed`` opcional → filtro de
    tamaño. Para K-means se usa la modalidad de phasor que esté disponible (FLIM si hay,
    si no espectral).

    Parameters
    ----------
    intensidad : numpy.ndarray, shape (alto, ancho)
    g_flim, s_flim, g_esp, s_esp : numpy.ndarray, optional
        Coordenadas de phasor por píxel. Solo se usan si ``metodo="kmeans"``.
    metodo : {"umbral", "kmeans"}, optional
        Estrategia de separación señal/fondo. Por defecto ``"umbral"`` (Otsu): sobre datos
        sintéticos da mejor recall de detección e IoU comparable a FIMAP. ``"kmeans"`` usa
        también la firma de phasor: sube la precisión (descarta cuerpos difusos de materia
        orgánica) a costa de recall. Ver ``docs/RESULTADOS_FASE2.md``.
    separar : bool, optional
        Si es ``True`` aplica :func:`separar_contacto`. Por defecto ``True``.
    tam_min : int, optional
        Área mínima en px de una partícula (descarta ruido). Por defecto ``8``.
    tam_max : int or None, optional
        Área máxima en px; ``None`` no filtra por arriba. Útil para descartar cuerpos de
        materia orgánica muy grandes.
    cerrar : int, optional
        Radio del cierre morfológico. Por defecto ``1``.
    semilla : int, optional

    Returns
    -------
    numpy.ndarray of int, shape (alto, ancho)
        Imagen de labels reindexada ``1..n`` (``0`` = fondo).

    Raises
    ------
    ValueError
        Si ``metodo`` no es válido.
    """
    if metodo not in METODOS:
        raise ValueError(f"metodo debe ser uno de {METODOS}, no {metodo!r}")

    if metodo == "kmeans":
        g = g_flim if g_flim is not None else g_esp
        s = s_flim if s_flim is not None else s_esp
        labels = segmentar_kmeans(
            intensidad, g, s, tam_min=tam_min, cerrar=cerrar, semilla=semilla
        )
    else:
        labels = segmentar_umbral(intensidad, tam_min=tam_min, cerrar=cerrar)

    if separar and labels.max() > 0:
        labels = separar_contacto(labels > 0, intensidad)

    return _filtrar_por_tamano(labels, tam_min, tam_max)


def _filtrar_por_tamano(labels: np.ndarray, tam_min: int, tam_max: int | None) -> np.ndarray:
    """Descarta labels fuera de ``[tam_min, tam_max]`` px y reindexa ``1..n``."""
    from skimage.segmentation import relabel_sequential

    labels = np.asarray(labels, dtype=np.int32).copy()
    if labels.max() == 0:
        return labels
    ids, cuentas = np.unique(labels[labels > 0], return_counts=True)
    for etiqueta, area in zip(ids, cuentas):
        if area < tam_min or (tam_max is not None and area > tam_max):
            labels[labels == etiqueta] = 0
    return np.asarray(relabel_sequential(labels)[0], dtype=np.int32)


def restringir_a_mascara(
    labels: np.ndarray,
    mascara_celular: np.ndarray,
    *,
    solape_min: float = 0.5,
) -> np.ndarray:
    """Conserva solo las ROIs cuyo solape con ``mascara_celular`` supera ``solape_min``.

    Para muestras de fagocitos (monocitos / neutrófilos): la máscara celular delimita las
    células que fagocitaron microplástico; quedarse con las partículas dentro de ellas
    aísla la señal de NR-MP de la autofluorescencia del resto del campo (Park et al. 2020).

    Parameters
    ----------
    labels : numpy.ndarray of int
        Segmentación de partículas.
    mascara_celular : numpy.ndarray
        Máscara binaria (o de labels) de las células.
    solape_min : float, optional
        Fracción mínima del área de la ROI que debe caer dentro de la máscara.
        Por defecto ``0.5``.

    Returns
    -------
    numpy.ndarray of int
        Labels filtrados y reindexados ``1..n``.
    """
    from skimage.segmentation import relabel_sequential

    labels = np.asarray(labels, dtype=np.int32).copy()
    dentro = np.asarray(mascara_celular) > 0
    for etiqueta in np.unique(labels[labels > 0]):
        roi = labels == etiqueta
        if (roi & dentro).sum() / roi.sum() < solape_min:
            labels[roi] = 0
    return np.asarray(relabel_sequential(labels)[0], dtype=np.int32)
