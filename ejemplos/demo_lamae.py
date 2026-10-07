"""Prueba de ``io_crudo`` con las imágenes reales de LAMAE (``ejemplo_lamae/``).

Muestra: polietileno (PE) teñido con Nile Red, objetivo 63x, excitación 440 nm.

- ``PE-NileRed-63x-440.sdt``        → FLIM (1024 bins, 59,96 MHz)
- ``PE-NileRed-63x-lmada-440.czi``  → λ-stack espectral (28 canales, 451-721 nm)
- ``PE-NileRed-63x-zstack-440.czi`` → z-stack (no espectral; no da phasor)

Las dos modalidades son adquisiciones **independientes** (512x512 vs 440x440, sin
registro espacial), así que cada una se procesa por separado. El ``.sdt`` trae **dos
detectores**; se usa el de más fotones. El FLIM se calibra con la IRF estimada del flanco
de subida (como SPCImage, :mod:`napari_mp_classifier.calibracion_flim`). No hay set de
calibración de los 6 polímeros, con lo cual esto no clasifica: ubica a PE en cada plano de
phasores y ejercita segmentación + features sobre datos reales. El informe completo de
caracterización lo genera ``ejemplos/informe_lamae.py``.

Uso::

    conda run -n napari-mp-env python ejemplos/demo_lamae.py   # o env base

Salida en ``ejemplos/salida_demo_lamae/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from napari_mp_classifier.features import extraer_features
from napari_mp_classifier.io_crudo import (
    listar_detectores_sdt,
    phasores_desde_czi,
    phasores_desde_sdt,
)
from napari_mp_classifier.segmentacion import segmentar

DATOS = RAIZ / "ejemplo_lamae"
SALIDA = RAIZ / "ejemplos" / "salida_demo_lamae"
SALIDA.mkdir(parents=True, exist_ok=True)


def _centro_ponderado(g, s, intensidad, pct=90.0):
    """Phasor promedio de los píxeles más brillantes (pondera por intensidad)."""
    umbral = np.nanpercentile(intensidad, pct)
    m = (intensidad >= umbral) & np.isfinite(g) & np.isfinite(s)
    w = intensidad[m]
    return float(np.average(g[m], weights=w)), float(np.average(s[m], weights=w)), int(m.sum())


def _figura_phasor(g, s, intensidad, titulo, ruta, *, modalidad="flim", pct=90.0):
    """Plano de phasores estilo SPCImage: semicírculo universal + nube de densidad fina.

    - Umbral de intensidad para quedarse con los píxeles de la partícula.
    - Nube: histograma 2D fino en escala log (no marcadores por píxel).
    - Marcador chico ('+') en el centroide ponderado por intensidad.
    """
    umbral = np.nanpercentile(intensidad, pct)
    m = (intensidad >= umbral) & np.isfinite(g) & np.isfinite(s)
    gc, sc, n = _centro_ponderado(g, s, intensidad, pct)

    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    if modalidad == "flim":
        rango = [[0.0, 1.0], [0.0, 0.62]]
        t = np.linspace(0, np.pi, 400)
        ax.plot(0.5 + 0.5 * np.cos(t), 0.5 * np.sin(t), "-", color="0.15", lw=1.3)
        bins = (400, 248)
    else:
        lim = 1.02
        rango = [[-lim, lim], [-lim, lim]]
        t = np.linspace(0, 2 * np.pi, 400)
        ax.plot(np.cos(t), np.sin(t), "-", color="0.15", lw=1.0)
        bins = (440, 440)

    h, xe, ye = np.histogram2d(g[m], s[m], bins=bins, range=rango)
    h = np.ma.masked_where(h == 0, h)
    ax.pcolormesh(xe, ye, h.T, norm=LogNorm(), cmap="turbo", shading="flat", rasterized=True)

    ax.plot(gc, sc, "+", ms=10, mew=1.8, color="k")
    ax.text(0.98, 0.96, f"centroide  g={gc:.3f}  s={sc:.3f}\npíxeles n={n:,}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox={"boxstyle": "round", "fc": "white", "ec": "0.7", "alpha": 0.9})
    ax.axhline(0, color="0.7", lw=0.5)
    ax.set_xlabel("g")
    ax.set_ylabel("s")
    ax.set_title(titulo, fontsize=10)
    ax.set_xlim(rango[0])
    ax.set_ylim(rango[1])
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    return gc, sc


def procesar_flim():
    print("\n--- FLIM (.sdt) ---")
    ruta = DATOS / "PE-NileRed-63x-440.sdt"
    for d in listar_detectores_sdt(ruta):
        print(f"detector {d['indice']}: {d['fotones'] / 1e6:.1f} M fotones")
    ph = phasores_desde_sdt(ruta)
    print(f"forma {ph.forma}  frecuencia {ph.frecuencia_mhz:.2f} MHz  bins {ph.metadatos['n_bins']}  "
          f"detector {ph.metadatos['detector']}  calibrado={ph.calibrado}")
    gc, sc = _figura_phasor(
        ph.g, ph.s, ph.intensidad,
        "phasor FLIM — crudo (IRF no removida)",
        SALIDA / "phasor_flim_crudo.png", modalidad="flim",
    )
    print(f"phasor PE crudo: g={gc:.3f}  s={sc:.3f}  -> fuera del semicírculo (desfase IRF)")

    ph_c = phasores_desde_sdt(ruta, calibrar_irf=True)
    gc2, sc2 = _figura_phasor(
        ph_c.g, ph_c.s, ph_c.intensidad,
        "phasor FLIM — calibrado con IRF del flanco (SPCImage)",
        SALIDA / "phasor_flim_irf.png", modalidad="flim",
    )
    from phasorpy.lifetime import phasor_to_apparent_lifetime

    tphi, tmod = phasor_to_apparent_lifetime(gc2, sc2, ph_c.frecuencia_mhz)
    print(f"phasor PE (calibrado IRF): g={gc2:.3f}  s={sc2:.3f}  "
          f"tau_phi={float(tphi):.2f} ns  tau_mod={float(tmod):.2f} ns")
    print("  [IRF estimada del flanco; con una IRF medida en la sesión se elimina la aproximación]")
    return ph


def procesar_espectral():
    print("\n--- Espectral (.czi λ-stack) ---")
    ph = phasores_desde_czi(DATOS / "PE-NileRed-63x-lmada-440.czi")
    r = ph.metadatos["rango_nm"]
    print(f"forma {ph.forma}  canales {ph.metadatos['n_canales']}  "
          f"rango {r[0]:.0f}-{r[1]:.0f} nm  calibrado={ph.calibrado}")
    gc, sc = _figura_phasor(
        ph.g, ph.s, ph.intensidad,
        "phasor espectral — PE / Nile Red (λ absoluta)",
        SALIDA / "phasor_espectral.png", modalidad="espectral",
    )
    ang = np.arctan2(sc, gc) % (2 * np.pi)
    ini, fin = r
    paso = (fin - ini) / (ph.metadatos["n_canales"] - 1)
    periodo = fin - ini + paso
    lambda_centro = ini + (ang / (2 * np.pi)) * periodo
    print(f"phasor PE espectral: g={gc:.3f}  s={sc:.3f}  ángulo={np.degrees(ang):.1f}°")
    print(f"  -> centro espectral aparente ≈ {lambda_centro:.0f} nm "
          f"(Nile Red en matriz apolar/hidrofóbica; coherente con PE)")
    return ph


def segmentar_y_features(ph_esp):
    print("\n--- Segmentación + features sobre la imagen espectral ---")
    # La adquisición es de una sola partícula que llena el campo: no separar por
    # watershed (partiría el fragmento en trozos artificiales).
    labels = segmentar(ph_esp.intensidad, metodo="umbral", separar=False, tam_min=50)
    n = int(labels.max())
    print(f"{n} ROIs detectadas")
    if n == 0:
        return
    feats = extraer_features(
        labels, ph_esp.intensidad, g_esp=ph_esp.g, s_esp=ph_esp.s,
        escala_um_px=0.0767,  # de coord_scales del .czi
    )
    ruta_csv = SALIDA / "features_espectral.csv"
    feats.to_csv(ruta_csv)
    print(f"features → {ruta_csv.relative_to(RAIZ)}")
    with np.printoptions(precision=3):
        print(feats[["area_px", "area_um2", "g_esp", "s_esp", "dispersion_esp"]]
              .describe().loc[["mean", "std", "min", "max"]])

    fig, axs = plt.subplots(1, 2, figsize=(11, 5.4))
    axs[0].imshow(ph_esp.intensidad, cmap="gray")
    axs[0].set_title("intensidad (suma espectral)")
    axs[1].imshow(labels, cmap="nipy_spectral")
    axs[1].set_title(f"segmentación — {n} ROIs")
    for a in axs:
        a.set_xticks([])
        a.set_yticks([])
    fig.tight_layout()
    fig.savefig(SALIDA / "segmentacion_espectral.png", dpi=130)
    plt.close(fig)


def main():
    print("=" * 74)
    print("PRUEBA io_crudo — imágenes reales LAMAE (PE / Nile Red 63x, exc 440)")
    print("=" * 74)
    procesar_flim()
    ph_esp = procesar_espectral()
    segmentar_y_features(ph_esp)

    print("\n--- z-stack (.czi no espectral) ---")
    try:
        phasores_desde_czi(DATOS / "PE-NileRed-63x-zstack-440.czi")
    except ValueError as e:
        print(f"rechazado como se espera: {e}")

    print(f"\nSalida en {SALIDA.relative_to(RAIZ)}/")


if __name__ == "__main__":
    main()
