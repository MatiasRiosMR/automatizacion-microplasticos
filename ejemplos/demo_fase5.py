"""Demo de la Fase 5: robustez del clasificador y flujo de fagocitos.

Experimentos sobre datos sintéticos (la validación con muestras reales queda pendiente de
los `.sdt`/`.czi` del equipo):

1. **Desajuste de envejecimiento**: la calibración usa el estándar (abrasión + H2O2 [+ UV]);
   se barre el grado de envejecimiento de la muestra por encima/por debajo de ese estándar.
2. **Ruido de adquisición**: se barre el σ del phasor de la muestra.
3. **Punto de operación de `confianza`**: para un desajuste moderado, se busca el valor que
   maximiza F1 sin perder polímero real.
4. **Fusión vs. una sola modalidad** bajo desajuste de envejecimiento.
5. **Desmezcla integrada**: muestra con mucha materia orgánica, con y sin enmascarado previo.
6. **Flujo de fagocitos**: `restringir_a_mascara` con una máscara celular sintética.

Uso::

    python ejemplos/demo_fase5.py

Salida en ``ejemplos/salida_demo_fase5/``.
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
    generar_mascara_celular,
    generar_particulas,
)

from napari_mp_classifier import (
    Calibracion,
    ClasificadorPhasor,
    analizar_muestra,
)
from napari_mp_classifier.desmezcla import (
    enmascarar_por_fraccion,
    fracciones_dos_componentes,
    phasor_mp_de_calibracion,
)
from napari_mp_classifier.metricas import evaluar_clasificacion
from napari_mp_classifier.segmentacion import restringir_a_mascara, segmentar

SALIDA = RAIZ / "ejemplos" / "salida_demo_fase5"


def _clf(modalidad, estrategia="knn", confianza=0.99, semilla=0):
    df = generar_calibracion(modalidad, n_por_polimero=80, sigma=0.02, semilla=semilla)
    cal = Calibracion.desde_dataframe(df, columnas=_columnas(modalidad))
    clf = ClasificadorPhasor(cal, estrategia=estrategia, confianza=confianza)
    if estrategia == "knn":
        clf.entrenar(df[_columnas(modalidad)].to_numpy(), df["polimero"].to_numpy())
    else:
        clf.entrenar()
    return clf


def _metricas(clf, X, y):
    pred = clf.predecir(X)
    pol = y != "no_clasificable"
    rep = evaluar_clasificacion(y[pol], pred[pol], incluir_no_clasificable=False)
    perdido = float((pred[pol] == "no_clasificable").mean())
    rech_org = float((pred[~pol] == "no_clasificable").mean()) if (~pol).any() else float("nan")
    return rep.exactitud, rep.f1_macro, perdido, rech_org


def exp_envejecimiento():
    print("\n1) DESAJUSTE DE ENVEJECIMIENTO (calibración = estándar abrasión+H2O2)")
    print(f"{'grado':>7}{'exactitud':>11}{'F1':>8}{'polímero perdido':>18}{'rechazo org':>13}")
    clf = _clf("fusion")
    filas = []
    for grado in (-0.20, -0.10, 0.0, 0.05, 0.10, 0.15, 0.20, 0.30):
        X, y = generar_particulas("fusion", n_por_polimero=80, n_no_clasificables=80,
                                  grado_envejecimiento=grado, semilla=5)
        ex, f1, perd, rech = _metricas(clf, X, y)
        filas.append((grado, ex, f1, perd, rech))
        print(f"{grado:>7.2f}{ex:>11.3f}{f1:>8.3f}{perd:>18.3f}{rech:>13.3f}")
    return filas


def exp_ruido():
    print("\n2) RUIDO DE ADQUISICIÓN (σ del phasor de la muestra)")
    print(f"{'sigma':>7}{'exactitud':>11}{'F1':>8}{'polímero perdido':>18}")
    clf = _clf("fusion")
    for sigma in (0.015, 0.025, 0.04, 0.06, 0.09):
        X, y = generar_particulas("fusion", n_por_polimero=80, n_no_clasificables=80,
                                  sigma=sigma, semilla=6)
        ex, f1, perd, _ = _metricas(clf, X, y)
        print(f"{sigma:>7.3f}{ex:>11.3f}{f1:>8.3f}{perd:>18.3f}")


def exp_confianza():
    print("\n3) PUNTO DE OPERACIÓN DE 'confianza' (desajuste de envejecimiento = 0.12)")
    print(f"{'confianza':>10}{'F1 (todas)':>12}{'polímero perdido':>18}{'rechazo org':>13}")
    mejor = (None, -1.0)
    for confianza in (0.90, 0.95, 0.975, 0.99, 0.995, 0.999):
        clf = _clf("fusion", confianza=confianza)
        X, y = generar_particulas("fusion", n_por_polimero=80, n_no_clasificables=80,
                                  grado_envejecimiento=0.12, semilla=7)
        pred = clf.predecir(X)
        rep = evaluar_clasificacion(y, pred)
        pol = y != "no_clasificable"
        perd = float((pred[pol] == "no_clasificable").mean())
        rech = float((pred[~pol] == "no_clasificable").mean())
        print(f"{confianza:>10.3f}{rep.f1_macro:>12.3f}{perd:>18.3f}{rech:>13.3f}")
        if rep.f1_macro > mejor[1]:
            mejor = (confianza, rep.f1_macro)
    print(f"   -> mejor F1 con confianza = {mejor[0]}")


def exp_fusion_vs_modalidad():
    print("\n4) FUSIÓN vs. UNA SOLA MODALIDAD bajo desajuste de envejecimiento")
    print(f"{'grado':>7}{'FLIM':>9}{'espectral':>12}{'fusión':>9}")
    clfs = {m: _clf(m) for m in ("flim", "espectral", "fusion")}
    for grado in (0.0, 0.05, 0.10, 0.15, 0.20):
        exac = {}
        for m in ("flim", "espectral", "fusion"):
            X, y = generar_particulas(m, n_por_polimero=80, n_no_clasificables=0,
                                      grado_envejecimiento=grado, semilla=8)
            exac[m] = float((clfs[m].predecir(X) == y).mean())
        print(f"{grado:>7.2f}{exac['flim']:>9.3f}{exac['espectral']:>12.3f}{exac['fusion']:>9.3f}")


def exp_desmezcla_integrada():
    print("\n5) DESMEZCLA INTEGRADA — muestra con mucha materia orgánica")
    canales, verdad = generar_imagen_muestra(n_materia_organica=20, semilla=13)
    df = generar_calibracion("fusion", n_por_polimero=80, semilla=0)
    cal = Calibracion.desde_dataframe(df, columnas=_columnas("fusion"))
    med = (df[_columnas("fusion")].to_numpy(), df["polimero"].to_numpy())

    p_mp = phasor_mp_de_calibracion(cal, "esp")
    cf = np.mean(list(centroides_referencia("fusion").values()), axis=0) + 0.22
    frac = fracciones_dos_componentes(canales["g_esp"], canales["s_esp"], p_mp, (cf[2], cf[3]))
    mascara_mp = enmascarar_por_fraccion(frac, umbral=0.4)

    for etiqueta, canales_usar in (("sin desmezcla", canales),
                                   ("con desmezcla", {**canales, "intensidad": canales["intensidad"] * mascara_mp})):
        res = analizar_muestra(canales_usar, cal, estrategia="knn", mediciones_calibracion=med, verdad=verdad)
        f = res.features
        org = f["polimero_real"] == "no_clasificable"
        pol = ~org
        exac = float((f.loc[pol, "polimero_predicho"] == f.loc[pol, "polimero_real"]).mean())
        falsos_mp = int(((f["polimero_real"] == "no_clasificable") & (f["polimero_predicho"] != "no_clasificable")).sum())
        print(f"  {etiqueta:<15} ROIs={res.n_rois:>3} ({int(org.sum())} de mat. orgánica)  "
              f"exactitud polímero={exac:.3f}  falsos 'MP' de mat. orgánica={falsos_mp}")


def exp_fagocitos():
    print("\n6) FLUJO DE FAGOCITOS — restringir_a_mascara")
    canales, verdad = generar_imagen_muestra(semilla=21)
    mascara = generar_mascara_celular(canales["intensidad"].shape, verdad,
                                      fraccion_fagocitada=0.6, semilla=21)
    labels = segmentar(canales["intensidad"], g_flim=canales["g_flim"], s_flim=canales["s_flim"])
    dentro = restringir_a_mascara(labels, mascara)
    print(f"  ROIs segmentadas: {labels.max()}  ->  dentro de células: {dentro.max()} "
          f"(máscara cubre {mascara.mean() * 100:.1f}% del campo)")


def main() -> None:
    SALIDA.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("DEMO FASE 5 — robustez y flujo de fagocitos (datos sintéticos)")
    print("=" * 78)

    filas = exp_envejecimiento()
    exp_ruido()
    exp_confianza()
    exp_fusion_vs_modalidad()
    exp_desmezcla_integrada()
    exp_fagocitos()

    _figura_envejecimiento(filas)
    print(f"\nFigura en {SALIDA / 'robustez_envejecimiento.png'}")


def _figura_envejecimiento(filas) -> None:
    import matplotlib.pyplot as plt

    from napari_mp_classifier.reportes import ESTILO_PUBLICACION

    grado, exac, _f1, perd, rech = (np.array(c) for c in zip(*filas))
    with plt.rc_context(ESTILO_PUBLICACION):
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        ax.axvline(0.0, color="#c3c2b7", lw=1.0)
        ax.plot(grado, exac, "o-", label="exactitud (polímeros)", color="#2a78d6")
        ax.plot(grado, perd, "s--", label="polímero perdido → no_clasificable", color="#eb6834")
        ax.plot(grado, rech, "^:", label="rechazo de materia orgánica", color="#1baf7a")
        ax.set_xlabel("desajuste de envejecimiento muestra − calibración")
        ax.set_ylabel("fracción")
        ax.set_ylim(-0.03, 1.05)
        ax.set_title("Robustez frente al desajuste de envejecimiento")
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
        fig.tight_layout()
        fig.savefig(SALIDA / "robustez_envejecimiento.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
