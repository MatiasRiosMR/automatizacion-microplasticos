# RESULTADOS_FASE3.md — pipeline completo, fusión, desmezcla, CLI

Reproducible con: `python ejemplos/demo_fase3.py` y `napari-mp-classifier classify`
(fecha de esta corrida: 2026-09-02). Estadísticos: media ± desvío sobre 12 imágenes
sintéticas.

## Qué agrega la Fase 3

| Módulo | Función |
|---|---|
| `pipeline.py` | `analizar_muestra(canales, calibracion, …)` → `ResultadoMuestra`: encadena segmentación → features → clasificación (+ métricas si hay verdad de terreno). Punto de entrada de librería. |
| `fusion.py` | `fusionar_por_roi` (empareja ROIs de dos segmentaciones registradas por centroide), `fusionar_por_decision` (combina dos clasificaciones independientes: acuerdo → polímero, desacuerdo/rechazo → `no_clasificable`). |
| `desmezcla.py` | `fracciones_dos_componentes` / `fracciones_multi_componente` (envuelven `phasorpy.component`), `enmascarar_por_fraccion`, `phasor_mp_de_calibracion`. Separa la fracción NR-MP de la autofluorescencia **antes** de clasificar. |
| `reportes.generar_reporte` | Informe unificado en una carpeta: `asignaciones.csv`, `resumen_muestra.md`, métricas, y figuras (phasores, overlay de segmentación, matriz de confusión, métricas por polímero). |
| `cli.py` | `napari-mp-classifier classify muestra.npz --calibracion cal.csv --salida out/` funcionando de punta a punta. |

## Pipeline completo (`analizar_muestra`, fusión 4D + knn)

| métrica | valor |
|---|---|
| IoU de segmentación | 0,813 ± 0,014 |
| exactitud de clasificación (todas las ROIs, incl. `no_clasificable`) | 0,952 ± 0,033 |
| exactitud sobre ROIs de **polímero** bien segmentadas | **0,994 ± 0,014** |
| rechazo de materia orgánica a nivel de ROI | 0,65 ± 0,28 |

Consistente con la Fase 2: el pipeline no degrada nada respecto a correr las etapas a
mano. El cuello de botella sigue siendo la segmentación y el rechazo de materia orgánica.

## Fusión 4D vs. fusión por decisión

Sobre ROIs sintéticas **bien segmentadas**, las tres vías (FLIM sola, espectral sola,
fusión 4D, fusión por decisión) dan prácticamente la misma exactitud (~0,94). Es
esperable y **no contradice** la Fase 1:

- El phasor por ROI es la **mediana espacial** de cientos de píxeles → su ruido efectivo
  es ~σ/√N, casi cero. Con clusters sintéticos tan limpios, 2D ya alcanza.
- La ventaja de la fusión aparece **bajo ruido / degradación**: en la Fase 1 (σ por
  partícula = 0,025) `fusion+knn` sacaba 0,987 vs. 0,88–0,94 de una modalidad sola, y en
  la Fase 5 se prueba con envejecimiento y desajuste de calibración.

`fusionar_por_decision` es la vía para cuando la `.sdt` y la `.czi` **no** están
registradas espacialmente (pregunta 9 de `docs/PREGUNTAS_DATOS.md`): no necesita
emparejar ROIs, solo combina las dos decisiones y es conservadora ante el desacuerdo.

## Desmezcla (fracción Nile Red-MP)

Con el componente NR-MP tomado como el centroide medio de los 6 polímeros y la
autofluorescencia como nube desplazada:

| región | fracción media NR-MP |
|---|---|
| partículas de polímero | **0,60** |
| materia orgánica / autofluorescencia | **0,13** |

La desmezcla **separa las dos poblaciones** (0,60 vs. 0,13): usar
`enmascarar_por_fraccion(frac, umbral≈0,4)` antes de segmentar debería recuperar buena
parte del rechazo de materia orgánica que el clasificador solo no logra de forma estable.
Se integra al pipeline en la Fase 5, cuando haya `.czi` reales con autofluorescencia
medida (hoy el phasor de autofluorescencia es sintético).

## CLI

```bash
napari-mp-classifier classify muestra.npz \
    --calibracion calibracion.csv --salida resultados/ \
    --estrategia knn --confianza 0.99 --escala-um-px 0.18
```

- `muestra.npz`: arrays 2D `intensidad` (obligatorio) + `g_flim`/`s_flim`/`g_esp`/`s_esp`
  (al menos un par). La modalidad se deduce sola.
- `calibracion.csv`: una fila por medición, con las columnas de phasor y `polimero`.
- Salida: la carpeta con el informe unificado de `generar_reporte`.
- Lectura de `.sdt`/`.czi` crudos: pendiente de `io_crudo.py` (datos reales del equipo).

## Limitaciones

- Todo sintético; ver `docs/RESULTADOS_FASE2.md`.
- El componente de autofluorescencia de la desmezcla es sintético (mean + 0,22), no medido.
- `fusionar_por_roi` no se probó con dos segmentaciones reales de distinta resolución
  (no hay datos); el test usa ROIs sintéticas con centroides conocidos.
