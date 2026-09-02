"""Extracción de features por ROI.

A partir de una imagen de ``labels`` (:mod:`napari_mp_classifier.segmentacion`) y de los
canales de la muestra (intensidad + coordenadas de phasor por píxel), calcula un
``DataFrame`` con una fila por ROI:

- **Phasor por ROI**: centro robusto (mediana espacial) de ``g``/``s`` FLIM y/o espectral
  sobre los píxeles de la ROI, vía :func:`phasorpy.phasor.phasor_center`. Es la feature
  que consume el clasificador.
- **Dispersión intra-ROI**: desvío de ``g``/``s`` dentro de la ROI — indicador de mezcla
  de materiales o de borde de contacto mal separado.
- **Intensidad**: total y media de Nile Red.
- **Forma** (``skimage.measure.regionprops``): área (px y µm²), excentricidad, solidez,
  extensión, relación de aspecto, perímetro, centroide.

:func:`matriz_features` arma el array ``X`` en el orden de columnas que espera
:class:`~napari_mp_classifier.clasificador.ClasificadorPhasor` para una modalidad dada.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Features de forma / intensidad que devuelve :func:`extraer_features` (además del phasor).
COLUMNAS_FORMA: tuple[str, ...] = (
    "area_px",
    "area_um2",
    "intensidad_total",
    "intensidad_media",
    "excentricidad",
    "solidez",
    "extension",
    "relacion_aspecto",
    "perimetro",
    "centro_fila",
    "centro_col",
)

_MODALIDAD_A_COLUMNAS: dict[str, list[str]] = {
    "flim": ["g_flim", "s_flim"],
    "espectral": ["g_esp", "s_esp"],
    "fusion": ["g_flim", "s_flim", "g_esp", "s_esp"],
}


def _centro_phasor(intensidad_roi: np.ndarray, g_roi: np.ndarray, s_roi: np.ndarray) -> tuple[float, float, float]:
    """Centro robusto (mediana espacial) de un conjunto de coordenadas de phasor.

    Envuelve :func:`phasorpy.phasor.phasor_center` con ``method="median"``. Devuelve
    ``(g, s, dispersion)`` donde ``dispersion`` es la media de los desvíos de ``g`` y ``s``.
    """
    from phasorpy.phasor import phasor_center

    finito = np.isfinite(g_roi) & np.isfinite(s_roi)
    if finito.sum() < 1:
        return float("nan"), float("nan"), float("nan")

    _, g_c, s_c = phasor_center(
        intensidad_roi[finito], g_roi[finito], s_roi[finito], method="median"
    )
    dispersion = float(np.nanmean([np.nanstd(g_roi[finito]), np.nanstd(s_roi[finito])]))
    return float(g_c), float(s_c), dispersion


def extraer_features(
    labels: np.ndarray,
    intensidad: np.ndarray,
    *,
    g_flim: np.ndarray | None = None,
    s_flim: np.ndarray | None = None,
    g_esp: np.ndarray | None = None,
    s_esp: np.ndarray | None = None,
    escala_um_px: float | None = None,
) -> pd.DataFrame:
    """Calcula las features de cada ROI.

    Parameters
    ----------
    labels : numpy.ndarray of int, shape (alto, ancho)
        Segmentación (``0`` = fondo). Cada valor ``> 0`` es una ROI.
    intensidad : numpy.ndarray, shape (alto, ancho)
        Imagen de intensidad de Nile Red.
    g_flim, s_flim, g_esp, s_esp : numpy.ndarray, shape (alto, ancho), optional
        Coordenadas de phasor por píxel. Se procesa cada par disponible
        (``g_flim``/``s_flim`` y ``g_esp``/``s_esp``).
    escala_um_px : float or None, optional
        Tamaño de píxel en µm para calcular ``area_um2``. ``None`` deja esa columna en
        ``NaN``.

    Returns
    -------
    pandas.DataFrame
        Una fila por ROI, indexada por el valor de label. Columnas:
        :data:`COLUMNAS_FORMA` + ``g_flim``/``s_flim``/``dispersion_flim`` y/o
        ``g_esp``/``s_esp``/``dispersion_esp`` según los canales pasados.

    Notes
    -----
    El phasor por ROI es la mediana espacial (robusta a píxeles de borde y a mezcla
    parcial con el fondo), no la media. La dispersión intra-ROI se reporta aparte porque
    un valor alto suele indicar que la ROI abarca más de un material o que ``watershed``
    no separó bien dos partículas en contacto.
    """
    from skimage.measure import regionprops

    labels = np.asarray(labels, dtype=int)
    intensidad = np.asarray(intensidad, dtype=float)
    pares = _pares_phasor(g_flim, s_flim, g_esp, s_esp)

    filas: list[dict] = []
    for region in regionprops(labels, intensity_image=intensidad):
        coords = region.coords
        fila_px, col_px = coords[:, 0], coords[:, 1]
        intensidad_roi = intensidad[fila_px, col_px]
        menor, mayor = region.axis_minor_length, region.axis_major_length

        fila: dict = {
            "label": region.label,
            "area_px": float(region.area),
            "area_um2": float(region.area) * escala_um_px**2 if escala_um_px else float("nan"),
            "intensidad_total": float(intensidad_roi.sum()),
            "intensidad_media": float(intensidad_roi.mean()),
            "excentricidad": float(region.eccentricity),
            "solidez": float(region.solidity),
            "extension": float(region.extent),
            "relacion_aspecto": float(mayor / menor) if menor > 0 else float("nan"),
            "perimetro": float(region.perimeter),
            "centro_fila": float(region.centroid[0]),
            "centro_col": float(region.centroid[1]),
        }

        for sufijo, (canal_g, canal_s) in pares.items():
            g_c, s_c, dispersion = _centro_phasor(
                intensidad_roi, canal_g[fila_px, col_px], canal_s[fila_px, col_px]
            )
            fila[f"g_{sufijo}"] = g_c
            fila[f"s_{sufijo}"] = s_c
            fila[f"dispersion_{sufijo}"] = dispersion

        filas.append(fila)

    columnas = ["label", *COLUMNAS_FORMA]
    for sufijo in pares:
        columnas += [f"g_{sufijo}", f"s_{sufijo}", f"dispersion_{sufijo}"]
    tabla = pd.DataFrame(filas, columns=columnas)
    return tabla.set_index("label")


def _pares_phasor(g_flim, s_flim, g_esp, s_esp) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Valida y arma los pares ``(g, s)`` disponibles, indexados por sufijo."""
    pares: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for sufijo, canal_g, canal_s in (("flim", g_flim, s_flim), ("esp", g_esp, s_esp)):
        if canal_g is None and canal_s is None:
            continue
        if canal_g is None or canal_s is None:
            raise ValueError(f"Hace falta g y s de la modalidad '{sufijo}', no solo una.")
        pares[sufijo] = (np.asarray(canal_g, dtype=float), np.asarray(canal_s, dtype=float))
    return pares


def matriz_features(
    features: pd.DataFrame, modalidad: str = "fusion"
) -> tuple[np.ndarray, list[str]]:
    """Extrae el array ``X`` de phasor por ROI en el orden que espera el clasificador.

    Parameters
    ----------
    features : pandas.DataFrame
        Salida de :func:`extraer_features`.
    modalidad : {"flim", "espectral", "fusion"}, optional
        Define qué columnas se toman y en qué orden. Por defecto ``"fusion"``.

    Returns
    -------
    X : numpy.ndarray, shape (n_rois, n_features)
    columnas : list of str
        Nombres de las columnas de ``X`` (para :class:`Calibracion` y los reportes).

    Raises
    ------
    ValueError
        Si ``modalidad`` no es válida o si faltan columnas en ``features``.
    """
    if modalidad not in _MODALIDAD_A_COLUMNAS:
        raise ValueError(
            f"modalidad debe ser 'flim', 'espectral' o 'fusion', no {modalidad!r}"
        )
    columnas = _MODALIDAD_A_COLUMNAS[modalidad]
    faltantes = [c for c in columnas if c not in features.columns]
    if faltantes:
        raise ValueError(
            f"El DataFrame de features no tiene las columnas {faltantes} "
            f"para la modalidad '{modalidad}'."
        )
    return features[columnas].to_numpy(dtype=float), list(columnas)
