"""Calibración del phasor FLIM con la IRF, al estilo SPCImage (Becker & Hickl).

Sin calibración, el phasor FLIM queda rotado y escalado por la **IRF** (función de
respuesta instrumental: pulso láser + detector + electrónica) y los tiempos de vida que se
leen de él son incorrectos. SPCImage no necesita un fluoróforo de referencia: usa la IRF,
medida o *sintética*, y corrige el phasor dividiéndolo por el phasor de la IRF::

    z_calibrado = z_medido / z_IRF        (z = g + i·s)

Eso equivale a :func:`phasorpy.lifetime.phasor_calibrate` con una referencia de tiempo de
vida 0, cuyo phasor teórico es ``(1, 0)``; no se reimplementa nada.

La IRF sintética se estima del **flanco de subida** del decaimiento: la derivada de
``IRF ∗ exp(−t/τ)`` es ``IRF − (IRF ∗ exp)/τ`` y, en el flanco, el primer término domina.

Validación (``docs/CALIBRACION_FLIM.md``): sobre decaimientos simulados a 59,96 MHz, 1024
canales y la ventana del TAC de LAMAE, el lifetime de fase recuperado tiene un error de
2–6 % entre 1 y 4 ns (rango de Nile Red), contra −14 % a −16 % de la corrección «pico → 0».
Por encima de ~5 ns el error crece por la ventana del TAC, no por la calibración.

Funciones
---------
- :func:`irf_desde_flanco` — IRF sintética a partir del decaimiento total.
- :func:`calibrar_con_irf` — calibra ``g``/``s`` por píxel con una IRF (medida o sintética).
- :func:`simular_decaimiento` — decaimiento sintético para validar la calibración.
"""

from __future__ import annotations

import numpy as np


def irf_desde_flanco(
    decaimiento: np.ndarray,
    *,
    suavizado: int = 3,
    ancho_max_bins: int | None = None,
    ventana_valida: tuple[int, int] | None = None,
) -> np.ndarray:
    """IRF sintética a partir del flanco de subida (método «auto IRF» de SPCImage).

    Parameters
    ----------
    decaimiento : numpy.ndarray, shape (n_bins,)
        Histograma temporal total (suma sobre los píxeles con señal).
    suavizado : int, optional
        Ancho (en canales) del promedio móvil previo a derivar, para no derivar ruido de
        Poisson. Por defecto ``3``.
    ancho_max_bins : int, optional
        Cuántos canales antes del pico se consideran flanco. Por defecto ~3 % del período.
    ventana_valida : tuple of int, optional
        ``(primer, último)`` canal con datos (los límites del TAC dejan ceros a los lados).
        Si es ``None`` se detecta sola.

    Returns
    -------
    numpy.ndarray, shape (n_bins,)
        IRF no negativa, normalizada a suma 1, en la misma grilla temporal.

    Raises
    ------
    ValueError
        Si el decaimiento no tiene un flanco de subida (p. ej. está vacío).
    """
    d = np.asarray(decaimiento, dtype=float)
    n = d.size
    if ventana_valida is None:
        nz = np.flatnonzero(d > 0)
        ventana_valida = (int(nz[0]), int(nz[-1])) if nz.size else (0, n - 1)
    k = max(1, int(suavizado))
    ds = np.convolve(d, np.ones(k) / k, mode="same")
    pico = int(np.argmax(ds))
    ancho = ancho_max_bins or max(8, int(0.03 * n))
    ini = max(ventana_valida[0] + k, pico - ancho, 1)
    derivada = np.zeros(n)
    derivada[ini:pico + 1] = np.diff(ds)[ini - 1:pico]
    derivada = np.clip(derivada, 0.0, None)
    if derivada.max() <= 0:
        raise ValueError("No se encontró un flanco de subida en el decaimiento.")
    # Lóbulo positivo que contiene el máximo de la derivada (descarta ruido previo).
    jmax = int(np.argmax(derivada))
    izq = jmax
    while izq > ini and derivada[izq - 1] > 0.02 * derivada[jmax]:
        izq -= 1
    irf = np.zeros(n)
    irf[izq:pico + 1] = derivada[izq:pico + 1]
    return irf / irf.sum()


def phasor_de_histograma(histograma: np.ndarray, armonico: int = 1) -> tuple[float, float, float]:
    """Phasor ``(media, g, s)`` de un único histograma temporal."""
    from phasorpy.phasor import phasor_from_signal

    m, g, s = phasor_from_signal(np.asarray(histograma, dtype=float), harmonic=armonico)
    return float(m), float(g), float(s)


def calibrar_con_irf(
    g: np.ndarray,
    s: np.ndarray,
    irf: np.ndarray,
    frecuencia_mhz: float,
    armonico: int = 1,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Corrige ``g``/``s`` con la IRF: ``z / z_IRF`` vía ``phasor_calibrate(lifetime=0)``.

    Parameters
    ----------
    g, s : numpy.ndarray
        Phasor FLIM sin calibrar (por píxel o por ROI).
    irf : numpy.ndarray, shape (n_bins,)
        IRF en la misma grilla temporal que la señal (medida o de :func:`irf_desde_flanco`).
    frecuencia_mhz : float
        Frecuencia de repetición del láser.
    armonico : int, optional
        Armónico del phasor. Por defecto ``1``.

    Returns
    -------
    g_cal, s_cal : numpy.ndarray
        Phasor calibrado.
    info : dict
        ``fase_irf_rad``, ``modulacion_irf``, ``g_irf``, ``s_irf``: la corrección aplicada.
    """
    from phasorpy.lifetime import phasor_calibrate

    m, gi, si = phasor_de_histograma(irf, armonico)
    g_cal, s_cal = phasor_calibrate(
        np.asarray(g, dtype=float), np.asarray(s, dtype=float), m, gi, si,
        frequency=frecuencia_mhz, lifetime=0.0, harmonic=armonico,
    )
    info = {"fase_irf_rad": float(np.arctan2(si, gi)), "modulacion_irf": float(np.hypot(gi, si)),
            "g_irf": gi, "s_irf": si}
    return np.asarray(g_cal), np.asarray(s_cal), info


def simular_decaimiento(
    tau_ns,
    frecuencia_mhz: float = 59.96,
    n_bins: int = 1024,
    *,
    t0_ns: float = 2.4,
    sigma_irf_ns: float = 0.09,
    cola_irf: float = 0.15,
    fotones: float = 1e6,
    ventana: tuple[int, int] = (40, 971),
    fracciones=None,
    semilla: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Decaimiento TCSPC sintético para validar la calibración.

    Decaimiento periódico (incluye la cola de los pulsos previos) convolucionado con una
    IRF gaussiana con cola de detector, recortado a la ventana del TAC y con ruido de
    Poisson. Los valores por defecto imitan la adquisición de LAMAE (59,96 MHz, 1024
    canales, límites del TAC 5 %–96 %).

    Parameters
    ----------
    tau_ns : float or sequence of float
        Tiempo(s) de vida en ns (multiexponencial si es una secuencia).
    frecuencia_mhz, n_bins : float, int, optional
        Frecuencia de repetición y número de canales.
    t0_ns, sigma_irf_ns, cola_irf : float, optional
        Posición, ancho y cola de la IRF simulada.
    fotones : float, optional
        Fotones totales del decaimiento.
    ventana : tuple of int, optional
        Canales válidos del TAC (fuera de la ventana el histograma es 0).
    fracciones : sequence of float, optional
        Amplitudes relativas de cada componente (por defecto iguales).
    semilla : int, optional

    Returns
    -------
    decaimiento : numpy.ndarray, shape (n_bins,)
    irf : numpy.ndarray, shape (n_bins,)
        IRF verdadera (normalizada a suma 1).
    """
    rng = np.random.default_rng(semilla)
    periodo = 1e3 / frecuencia_mhz
    t = np.arange(n_bins) * periodo / n_bins
    irf = np.exp(-0.5 * ((t - t0_ns) / sigma_irf_ns) ** 2)
    irf += cola_irf * np.where(t > t0_ns, np.exp(-(t - t0_ns) / 0.25), 0.0)
    irf /= irf.sum()
    taus = np.atleast_1d(np.asarray(tau_ns, dtype=float))
    fr = np.full(len(taus), 1.0 / len(taus)) if fracciones is None else np.asarray(fracciones, float)
    dec = sum(f * np.exp(-t / ta) / (1 - np.exp(-periodo / ta)) for f, ta in zip(fr, taus))
    senal = np.clip(np.real(np.fft.ifft(np.fft.fft(irf) * np.fft.fft(dec))), 0, None)
    senal = rng.poisson(senal * fotones / senal.sum()).astype(float)
    mascara = np.zeros(n_bins)
    mascara[ventana[0]:ventana[1] + 1] = 1.0
    return senal * mascara, irf
