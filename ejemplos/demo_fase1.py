"""Demo end-to-end de la Fase 1 sobre datos sintéticos.

Genera calibración + partículas sintéticas (6 polímeros + materia orgánica), clasifica con
las 3 estrategias y las 3 modalidades, e imprime las métricas estándar. Escribe un reporte
(CSVs + texto) y **figuras de calidad publicación/póster** a ``ejemplos/salida_demo/``:

- ``figuras/phasores_<modalidad>_<estrategia>.png|pdf`` — diagrama de phasores con los
  clusters de referencia (centroide + elipse de covarianza) y las partículas clasificadas.
- ``figuras/matriz_confusion_fusion_knn.png|pdf`` — mapa de calor normalizado por fila.
- ``figuras/metricas_por_polimero_fusion_knn.png|pdf`` — barras precisión/recall/F1.
- ``figuras/comparacion_modalidades.png|pdf`` — exactitud y F1 por modalidad × estrategia.

Uso::

    python ejemplos/demo_fase1.py
"""

from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "tests"))

import numpy as np  # noqa: E402

from napari_mp_classifier import Calibracion, ClasificadorPhasor  # noqa: E402
from napari_mp_classifier.metricas import evaluar_clasificacion  # noqa: E402
from napari_mp_classifier.reportes import (  # noqa: E402
    figura_comparacion,
    figura_matriz_confusion,
    figura_metricas_por_clase,
    figura_phasores,
    guardar_figura,
    guardar_reporte_metricas,
    resultados_a_dataframe,
)
from datos_sinteticos import _columnas, generar_calibracion, generar_particulas  # noqa: E402

MODALIDADES = ("flim", "espectral", "fusion")
ESTRATEGIAS = ("centroide", "knn", "gmm")


def correr(modalidad: str, estrategia: str, confianza: float | None = 0.99):
    columnas = _columnas(modalidad)

    df_cal = generar_calibracion(modalidad, n_por_polimero=60, sigma=0.02, semilla=0)
    cal = Calibracion.desde_dataframe(df_cal, columnas=columnas)

    clf = ClasificadorPhasor(cal, estrategia=estrategia, confianza=confianza)
    if estrategia == "knn":
        clf.entrenar(df_cal[columnas].to_numpy(), df_cal["polimero"].to_numpy())
    else:
        clf.entrenar()

    X, y = generar_particulas(
        modalidad, n_por_polimero=50, n_no_clasificables=80, sigma=0.025, semilla=42
    )
    pred, score = clf.predecir_con_score(X)
    rep = evaluar_clasificacion(y, pred)
    return cal, X, y, pred, score, rep, columnas


def main() -> None:
    print("=" * 78)
    print("DEMO FASE 1 — clasificador de microplásticos sobre phasores sintéticos")
    print("=" * 78)
    print(
        "Calibración: 60 mediciones/polímero (σ=0.02)\n"
        "Muestra:     50 partículas/polímero + 80 de 'materia orgánica' (σ=0.025)\n"
        "Regla no_clasificable: confianza=0.99 (umbral chi² consciente de la dimensión)\n"
    )

    tabla_resumen = []
    for modalidad in MODALIDADES:
        for estrategia in ESTRATEGIAS:
            _, _, y, pred, _, rep, _ = correr(modalidad, estrategia)
            sin_nc = evaluar_clasificacion(y, pred, incluir_no_clasificable=False)
            es_org = y == "no_clasificable"
            rechazo_org = float((pred[es_org] == "no_clasificable").mean())
            falso_nc = float((pred[~es_org] == "no_clasificable").mean())
            tabla_resumen.append(
                {
                    "modalidad": modalidad,
                    "estrategia": estrategia,
                    "exactitud": rep.exactitud,
                    "F1_macro": rep.f1_macro,
                    "F1_polimeros": sin_nc.f1_macro,
                    "rechazo_org": rechazo_org,
                    "falso_no_clasif": falso_nc,
                }
            )

    print(f"{'modalidad':<11}{'estrategia':<12}{'exact.':>8}{'F1(all)':>9}"
          f"{'F1(polim)':>11}{'rech.org':>10}{'falso_NC':>10}")
    print("-" * 78)
    for f in tabla_resumen:
        print(
            f"{f['modalidad']:<11}{f['estrategia']:<12}"
            f"{f['exactitud']:>8.3f}{f['F1_macro']:>9.3f}{f['F1_polimeros']:>11.3f}"
            f"{f['rechazo_org']:>10.3f}{f['falso_no_clasif']:>10.3f}"
        )

    # Reporte detallado para el caso ganador: fusión + knn (ver docs/RESULTADOS_FASE1.md).
    print("\n" + "=" * 78)
    print("DETALLE — modalidad=fusion, estrategia=knn")
    print("=" * 78)
    cal, X, y, pred, score, rep, columnas = correr("fusion", "knn")
    print(rep.resumen())

    carpeta = RAIZ / "ejemplos" / "salida_demo"
    carpeta_fig = carpeta / "figuras"
    guardar_reporte_metricas(rep, carpeta)
    tabla = resultados_a_dataframe(X, pred, score, columnas=columnas, ids=np.arange(len(X)))
    tabla.insert(1, "polimero_real", y)
    tabla.to_csv(carpeta / "asignaciones.csv", index=False)
    cal.guardar_csv(carpeta / "calibracion.csv")

    # ---------------------------------------------------------------- figuras
    print("\nGenerando figuras...")

    # Diagramas de phasores: una modalidad simple vs. la fusión (misma estrategia knn),
    # para que se vea que combinar FLIM + espectral separa lo que cada una sola confunde.
    for modalidad in ("flim", "espectral", "fusion"):
        cal_m, X_m, y_m, pred_m, _, rep_m, columnas_m = correr(modalidad, "knn")
        fig = figura_phasores(
            cal_m, X_m, pred_m, columnas_m, etiquetas_reales=y_m,
            resaltar_errores=True,
            titulo=f"Clasificación sobre phasores sintéticos — {modalidad} + knn "
                   f"(exactitud {rep_m.exactitud:.3f})",
        )
        guardar_figura(fig, carpeta_fig / f"phasores_{modalidad}_knn")

    fig = figura_matriz_confusion(
        rep, titulo="Matriz de confusión — fusión FLIM + espectral, knn"
    )
    guardar_figura(fig, carpeta_fig / "matriz_confusion_fusion_knn")

    fig = figura_metricas_por_clase(
        rep, titulo="Métricas por polímero — fusión FLIM + espectral, knn"
    )
    guardar_figura(fig, carpeta_fig / "metricas_por_polimero_fusion_knn")

    fig = figura_comparacion(tabla_resumen)
    guardar_figura(fig, carpeta_fig / "comparacion_modalidades")

    print(f"\nReporte escrito en: {carpeta}")
    for p in sorted(carpeta.rglob("*")):
        if p.is_file():
            print(f"  - {p.relative_to(carpeta)}")


if __name__ == "__main__":
    main()
