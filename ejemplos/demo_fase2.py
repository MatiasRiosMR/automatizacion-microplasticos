"""Demo end-to-end de la Fase 2 sobre una imagen sintética de muestra.

Genera una imagen de microscopía sintética (partículas de los 6 polímeros + cuerpos de
materia orgánica + fondo), la segmenta, extrae features por ROI, clasifica cada ROI contra
la calibración sintética de la Fase 1 y reporta:

- Métricas de **segmentación** (IoU, precisión/recall de detección) vs. la verdad de terreno.
- Métricas de **clasificación** de las ROIs bien segmentadas.
- Figuras en ``ejemplos/salida_demo_fase2/``:
  - ``segmentacion_<metodo>.png|pdf`` — imagen con las ROIs detectadas.
  - ``clasificacion_rois.png|pdf`` — imagen con las ROIs coloreadas por polímero predicho.
  - ``phasores_rois.png|pdf`` — diagrama de phasores con las ROIs sobre los clusters de ref.
  - ``matriz_confusion_rois.png|pdf`` — matriz de confusión de las ROIs.

Uso::

    python ejemplos/demo_fase2.py
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
    generar_calibracion,
    generar_imagen_muestra,
)

from napari_mp_classifier import Calibracion, ClasificadorPhasor
from napari_mp_classifier.features import extraer_features, matriz_features
from napari_mp_classifier.metricas import (
    emparejar_rois,
    evaluar_clasificacion,
    evaluar_segmentacion,
)
from napari_mp_classifier.reportes import (
    figura_matriz_confusion,
    figura_phasores,
    figura_segmentacion,
    guardar_figura,
    guardar_reporte_metricas,
    resultados_a_dataframe,
)
from napari_mp_classifier.segmentacion import segmentar

ESCALA_UM_PX = 0.18  # tamaño de píxel ilustrativo


def main() -> None:
    print("=" * 78)
    print("DEMO FASE 2 — segmentación + features + clasificación por ROI (sintético)")
    print("=" * 78)

    canales, verdad = generar_imagen_muestra(semilla=7)
    n_org = sum(1 for v in verdad["polimero"].values() if v == "no_clasificable")
    print(
        f"Imagen: {canales['intensidad'].shape}, "
        f"{len(verdad['polimero'])} ROIs verdaderas ({n_org} de materia orgánica).\n"
    )

    salida = RAIZ / "ejemplos" / "salida_demo_fase2"
    figs = salida / "figuras"

    # ---------------------------------------------------------------- segmentación
    print(f"{'método':<10}{'IoU':>8}{'P det.':>9}{'R det.':>9}{'F1 det.':>9}{'n ROIs':>9}")
    print("-" * 53)
    labels_por_metodo = {}
    for metodo in ("umbral", "kmeans"):
        labels = segmentar(
            canales["intensidad"],
            g_flim=canales["g_flim"], s_flim=canales["s_flim"],
            g_esp=canales["g_esp"], s_esp=canales["s_esp"],
            metodo=metodo,
        )
        labels_por_metodo[metodo] = labels
        rep_seg = evaluar_segmentacion(labels, verdad["labels"], iou_min=0.5)
        print(
            f"{metodo:<10}{rep_seg.iou_medio:>8.3f}{rep_seg.precision_deteccion:>9.3f}"
            f"{rep_seg.recall_deteccion:>9.3f}{rep_seg.f1_deteccion:>9.3f}"
            f"{rep_seg.n_predichas:>9d}"
        )
        fig = figura_segmentacion(
            canales["intensidad"], labels,
            titulo=f"Segmentación — método {metodo} "
                   f"(IoU {rep_seg.iou_medio:.2f}, recall {rep_seg.recall_deteccion:.2f})",
        )
        guardar_figura(fig, figs / f"segmentacion_{metodo}")

    labels = labels_por_metodo["umbral"]

    # ---------------------------------------------------------------- features
    feats = extraer_features(
        labels, canales["intensidad"],
        g_flim=canales["g_flim"], s_flim=canales["s_flim"],
        g_esp=canales["g_esp"], s_esp=canales["s_esp"],
        escala_um_px=ESCALA_UM_PX,
    )
    print(f"\nFeatures extraídas: {feats.shape[0]} ROIs × {feats.shape[1]} columnas.")
    print("Columnas:", ", ".join(feats.columns))

    # ---------------------------------------------------------------- clasificación
    df_cal = generar_calibracion("fusion", n_por_polimero=60, sigma=0.02, semilla=0)
    cal = Calibracion.desde_dataframe(df_cal, columnas=_columnas("fusion"))
    clf = ClasificadorPhasor(cal, estrategia="knn", confianza=0.99)
    clf.entrenar(df_cal[_columnas("fusion")].to_numpy(), df_cal["polimero"].to_numpy())

    X, columnas = matriz_features(feats, "fusion")
    pred, score = clf.predecir_con_score(X)

    # Verdad de terreno por ROI predicha (por solape con la segmentación verdadera).
    emparejadas = emparejar_rois(labels, verdad["labels"], iou_min=0.3)
    verd_por_pred = {lp: verdad["polimero"][lv] for lv, (lp, _) in emparejadas.items()}
    y_true = np.array([verd_por_pred.get(l, "no_clasificable") for l in feats.index])

    rep = evaluar_clasificacion(y_true, pred)
    print("\n" + "=" * 78)
    print("CLASIFICACIÓN DE LAS ROIs")
    print("=" * 78)
    print(rep.resumen())

    solo_polimero = y_true != "no_clasificable"
    if solo_polimero.any():
        rep_pol = evaluar_clasificacion(y_true[solo_polimero], pred[solo_polimero])
        print(f"\nExactitud solo sobre ROIs de polímero: {rep_pol.exactitud:.3f}")

    # ---------------------------------------------------------------- reporte a disco
    guardar_reporte_metricas(rep, salida)
    tabla = resultados_a_dataframe(X, pred, score, columnas=columnas, ids=feats.index.to_numpy())
    tabla.insert(1, "polimero_real", y_true)
    tabla = tabla.join(feats.drop(columns=columnas), on="id")
    tabla.to_csv(salida / "features_y_asignaciones.csv", index=False)

    fig = figura_segmentacion(
        canales["intensidad"], labels,
        etiquetas_por_label={int(l): p for l, p in zip(feats.index, pred)},
        titulo="ROIs clasificadas por polímero predicho",
    )
    guardar_figura(fig, figs / "clasificacion_rois")

    fig = figura_phasores(
        cal, X, pred, columnas, etiquetas_reales=y_true, resaltar_errores=True,
        titulo="Phasores de las ROIs sobre los clusters de referencia (fusión, knn)",
    )
    guardar_figura(fig, figs / "phasores_rois")

    fig = figura_matriz_confusion(rep, titulo="Matriz de confusión — ROIs de la muestra")
    guardar_figura(fig, figs / "matriz_confusion_rois")

    print(f"\nReporte escrito en: {salida}")
    for p in sorted(salida.rglob("*")):
        if p.is_file():
            print(f"  - {p.relative_to(salida)}")


if __name__ == "__main__":
    main()
