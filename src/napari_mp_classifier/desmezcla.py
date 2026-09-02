"""Desmezcla espectral: separar la fracción de Nile Red-MP de la autofluorescencia.

En matrices complejas —muestras ambientales con materia orgánica fluorescente, y
cultivos de monocitos / neutrófilos que fagocitaron microplástico— la señal medida en
cada píxel es una **mezcla** de la emisión de Nile Red unido al microplástico y de la
autofluorescencia del entorno (celulosa, quitina, restos celulares, NAD(P)H, flavinas).
Clasificar directamente esa mezcla lleva a falsos positivos.

Este módulo separa las contribuciones **antes** de clasificar, apoyándose en
:mod:`phasorpy.component` (unmixing basado en phasores, model-free, de Vitrani & Cutrale
2022; ver ``docs/SPECTRAL_UNMIXING.md``). No reimplementa el cálculo.

Flujo típico
------------
1. Estimar los phasores de referencia de los componentes: el de Nile Red-MP a partir de
   la :class:`~napari_mp_classifier.calibracion.Calibracion` (centroide medio de los 6
   polímeros, o el del polímero esperado), el de la autofluorescencia midiéndola en una
   zona sin microplástico o dejándola como componente desconocido.
2. :func:`fracciones_dos_componentes` (o :func:`fracciones_multi_componente`) → mapa de
   la fracción de NR-MP por píxel.
3. :func:`enmascarar_por_fraccion` → máscara de los píxeles dominados por NR-MP, que se
   pasa a :func:`~napari_mp_classifier.segmentacion.segmentar` o se usa para ponderar el
   phasor por ROI en :func:`~napari_mp_classifier.features.extraer_features`.

Desmezcla ≠ clasificación: son pasos distintos. La desmezcla dice *cuánto* de la señal es
microplástico; la clasificación dice *qué polímero* es.
"""

from __future__ import annotations

import numpy as np


def fracciones_dos_componentes(
    g: np.ndarray,
    s: np.ndarray,
    phasor_mp: tuple[float, float],
    phasor_autofluorescencia: tuple[float, float],
) -> np.ndarray:
    """Fracción de Nile Red-MP por píxel, con dos componentes conocidos.

    Proyecta cada coordenada de phasor sobre la recta que une los dos componentes y
    devuelve la distancia relativa al componente de autofluorescencia (envuelve
    :func:`phasorpy.component.phasor_component_fraction`).

    Parameters
    ----------
    g, s : numpy.ndarray
        Coordenadas de phasor por píxel (misma forma).
    phasor_mp : tuple of float
        Coordenadas ``(g, s)`` del componente Nile Red-MP.
    phasor_autofluorescencia : tuple of float
        Coordenadas ``(g, s)`` del componente de autofluorescencia.

    Returns
    -------
    numpy.ndarray
        Fracción de NR-MP en ``[0, 1]`` (recortada), misma forma que ``g``. Los píxeles
        con ``g``/``s`` no finito quedan en ``NaN``.
    """
    from phasorpy.component import phasor_component_fraction

    g = np.asarray(g, dtype=float)
    s = np.asarray(s, dtype=float)
    finito = np.isfinite(g) & np.isfinite(s)

    fraccion = np.full(g.shape, np.nan)
    if finito.any():
        valores = phasor_component_fraction(
            g[finito], s[finito],
            [phasor_mp[0], phasor_autofluorescencia[0]],
            [phasor_mp[1], phasor_autofluorescencia[1]],
        )
        fraccion[finito] = np.clip(valores, 0.0, 1.0)
    return fraccion


def fracciones_multi_componente(
    intensidad: np.ndarray,
    g: np.ndarray,
    s: np.ndarray,
    componentes: dict[str, tuple[float, float]],
) -> dict[str, np.ndarray]:
    """Fracción de cada componente por píxel, con N componentes conocidos.

    Envuelve :func:`phasorpy.component.phasor_component_fit` (solución por mínimos
    cuadrados; hasta 3 componentes con un solo armónico). Útil cuando la autofluorescencia
    tiene más de una firma (p. ej. NAD(P)H + flavinas + NR-MP).

    Parameters
    ----------
    intensidad : numpy.ndarray
        Intensidad por píxel.
    g, s : numpy.ndarray
        Coordenadas de phasor por píxel (misma forma que ``intensidad``).
    componentes : dict[str, tuple of float]
        Nombre → coordenadas ``(g, s)`` de cada componente de referencia.

    Returns
    -------
    dict[str, numpy.ndarray]
        Nombre → mapa de fracción (misma forma que ``intensidad``), con ``NaN`` donde el
        phasor no es finito.
    """
    from phasorpy.component import phasor_component_fit

    intensidad = np.asarray(intensidad, dtype=float)
    g = np.asarray(g, dtype=float)
    s = np.asarray(s, dtype=float)
    nombres = list(componentes)
    comp_g = [componentes[n][0] for n in nombres]
    comp_s = [componentes[n][1] for n in nombres]

    finito = np.isfinite(g) & np.isfinite(s) & np.isfinite(intensidad)
    fracciones = {n: np.full(intensidad.shape, np.nan) for n in nombres}
    if finito.any():
        resultado = phasor_component_fit(
            intensidad[finito], g[finito], s[finito], comp_g, comp_s
        )
        resultado = np.atleast_2d(resultado)
        if resultado.shape[0] != len(nombres):
            resultado = resultado.T
        for i, nombre in enumerate(nombres):
            fracciones[nombre][finito] = resultado[i]
    return fracciones


def enmascarar_por_fraccion(
    fraccion_mp: np.ndarray,
    *,
    umbral: float = 0.5,
) -> np.ndarray:
    """Máscara booleana de los píxeles dominados por Nile Red-MP.

    Parameters
    ----------
    fraccion_mp : numpy.ndarray
        Mapa de fracción de NR-MP (salida de :func:`fracciones_dos_componentes`).
    umbral : float, optional
        Fracción mínima para considerar el píxel "microplástico". Por defecto ``0.5``.

    Returns
    -------
    numpy.ndarray of bool
        ``True`` donde ``fraccion_mp >= umbral`` (los ``NaN`` dan ``False``).
    """
    fraccion_mp = np.asarray(fraccion_mp, dtype=float)
    return np.nan_to_num(fraccion_mp, nan=0.0) >= umbral


def phasor_mp_de_calibracion(calibracion, modalidad: str = "esp") -> tuple[float, float]:
    """Phasor de referencia del componente Nile Red-MP a partir de una calibración.

    Devuelve el centroide medio de los 6 polímeros en el plano de phasores de la
    modalidad pedida — una aproximación razonable del componente "microplástico teñido"
    cuando no se conoce el polímero de antemano.

    Parameters
    ----------
    calibracion : Calibracion
        Debe tener columnas de la modalidad pedida (``g_esp``/``s_esp`` o
        ``g_flim``/``s_flim``), o ser una calibración de fusión (4D).
    modalidad : {"esp", "flim"}, optional
        Modalidad de la que tomar el phasor. Por defecto ``"esp"``.

    Returns
    -------
    tuple of float
        Coordenadas ``(g, s)`` del componente.
    """
    columnas = list(calibracion.columnas)
    par = ("g_esp", "s_esp") if modalidad == "esp" else ("g_flim", "s_flim")
    if par[0] not in columnas:
        raise ValueError(
            f"La calibración no tiene la modalidad '{modalidad}' (columnas: {columnas})."
        )
    ig, is_ = columnas.index(par[0]), columnas.index(par[1])
    centros = calibracion.matriz_centroides()
    return float(centros[:, ig].mean()), float(centros[:, is_].mean())
