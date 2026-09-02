"""Demo end-to-end de la Fase 3: pipeline completo + fusión + desmezcla + reporte unificado.

1. `analizar_muestra` (segmentación → features → clasificación) sobre una imagen sintética
   de muestra, con `generar_reporte` escribiendo el informe completo.
2. Comparación **fusión 4D** (un clasificador en `[g_flim, s_flim, g_esp, s_esp]`) vs.
   **fusión por decisión** (clasificar cada modalidad y combinar).
3. Desmezcla: mapa de fracción Nile Red-MP vs. autofluorescencia (`phasorpy.component`).

Uso::

    python ejemplos/demo_fase3.py

Salida en ``ejemplos/salida_demo_fase3/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "tests"))

import numpy as np
from datos_sinteticos import (
    _columnas,
    centroides_referencia,
    generar_calibracion,
    generar_imagen_muestra,
)

from napari_mp_classifier import (
    Calibracion,
    ClasificadorPhasor,
    analizar_muestra,
)
from napari_mp_classifier.desmezcla import (
    fracciones_dos_componentes,
    phasor_mp_de_calibracion,
)
from napari_mp_classifier.features import extraer_features, matriz_features
from napari_mp_classifier.fusion import fusionar_por_decision
from napari_mp_classifier.metricas import emparejar_rois, evaluar_clasificacion
from napari_mp_classifier.reportes import (
    figura_segmentacion,
    generar_reporte,
    guardar_figura,
)
from napari_mp_classifier.segmentacion import segmentar

SALIDA = RAIZ / "ejemplos" / "salida_demo_fase3"


def _calibracion(modalidad):
    df = generar_calibracion(modalidad, n_por_polimero=60, sigma=0.02, semilla=0)
    cal = Calibracion.desde_dataframe(df, columnas=_columnas(modalidad))
    return cal, (df[_columnas(modalidad)].to_numpy(), df["polimero"].to_numpy())


def _verdad_por_roi(labels, feats, verdad):
    emp = emparejar_rois(labels, verdad["labels"], iou_min=0.3)
    verd_por_pred = {lp: verdad["polimero"][lv] for lv, (lp, _) in emp.items()}
    return np.array([verd_por_pred.get(l, "no_clasificable") for l in feats.index])


def main() -> None:
    print("=" * 78)
    print("DEMO FASE 3 — pipeline completo + fusión + desmezcla")
    print("=" * 78)

    canales, verdad = generar_imagen_muestra(semilla=11)

    # ---------------------------------------------------------- 1) pipeline + reporte
    cal_fus, med_fus = _calibracion("fusion")
    resultado = analizar_muestra(
        canales, cal_fus, estrategia="knn", mediciones_calibracion=med_fus,
        escala_um_px=0.18, verdad=verdad,
    )
    print(f"\nPipeline: {resultado.n_rois} ROIs.")
    print(resultado.reporte_segmentacion.resumen())
    print("\n" + resultado.reporte_clasificacion.resumen())

    rutas = generar_reporte(resultado, SALIDA, canales=canales, titulo="muestra_sintetica")
    print(f"\nInforme unificado en {SALIDA}:")
    for clave, ruta in rutas.items():
        print(f"  - {clave}: {ruta.relative_to(SALIDA)}")

    # ---------------------------------------------- 2) fusión 4D vs. fusión por decisión
    print("\n" + "=" * 78)
    print("FUSIÓN 4D vs. FUSIÓN POR DECISIÓN")
    print("=" * 78)
    labels = segmentar(
        canales["intensidad"],
        g_flim=canales["g_flim"], s_flim=canales["s_flim"],
        g_esp=canales["g_esp"], s_esp=canales["s_esp"],
    )
    feats = extraer_features(
        labels, canales["intensidad"],
        g_flim=canales["g_flim"], s_flim=canales["s_flim"],
        g_esp=canales["g_esp"], s_esp=canales["s_esp"],
    )
    y_true = _verdad_por_roi(labels, feats, verdad)

    preds = {}
    for modalidad in ("flim", "espectral", "fusion"):
        cal, med = _calibracion(modalidad)
        clf = ClasificadorPhasor(cal, estrategia="knn", confianza=0.99).entrenar(*med)
        X, _ = matriz_features(feats, modalidad)
        preds[modalidad] = clf.predecir(X)

    pred_decision = fusionar_por_decision(preds["flim"], preds["espectral"])

    print(f"{'estrategia':<22}{'exactitud':>12}{'F1 macro':>12}")
    print("-" * 46)
    for nombre, pred in [
        ("FLIM sola", preds["flim"]),
        ("espectral sola", preds["espectral"]),
        ("fusión 4D", preds["fusion"]),
        ("fusión por decisión", pred_decision),
    ]:
        rep = evaluar_clasificacion(y_true, pred)
        print(f"{nombre:<22}{rep.exactitud:>12.3f}{rep.f1_macro:>12.3f}")

    # ------------------------------------------------------------ 3) desmezcla
    print("\n" + "=" * 78)
    print("DESMEZCLA — fracción Nile Red-MP vs. autofluorescencia")
    print("=" * 78)
    p_mp = phasor_mp_de_calibracion(cal_fus, "esp")
    cf = np.mean(list(centroides_referencia("fusion").values()), axis=0) + 0.22
    p_auto = (cf[2], cf[3])
    frac = fracciones_dos_componentes(canales["g_esp"], canales["s_esp"], p_mp, p_auto)

    es_pol = np.zeros(labels.shape, dtype=bool)
    es_org = np.zeros(labels.shape, dtype=bool)
    for etiqueta, codigo in verdad["polimero"].items():
        (es_org if codigo == "no_clasificable" else es_pol)[verdad["labels"] == etiqueta] = True
    print(f"Fracción media NR-MP en partículas de polímero : {np.nanmean(frac[es_pol]):.2f}")
    print(f"Fracción media NR-MP en materia orgánica       : {np.nanmean(frac[es_org]):.2f}")

    import matplotlib.pyplot as plt

    with plt.rc_context({"figure.dpi": 120}):
        fig, ax = plt.subplots(figsize=(6.4, 6.0))
        im = ax.imshow(np.nan_to_num(frac, nan=0.0), cmap="magma", vmin=0, vmax=1)
        ax.set_title("Fracción de Nile Red-MP por píxel")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
    guardar_figura(fig, SALIDA / "figuras" / "fraccion_nr_mp")

    fig = figura_segmentacion(canales["intensidad"], labels, titulo="Segmentación (referencia)")
    guardar_figura(fig, SALIDA / "figuras" / "segmentacion_referencia")

    print(f"\nFiguras de desmezcla en {SALIDA / 'figuras'}")


if __name__ == "__main__":
    main()
