# RESULTADOS_FASE1.md — prueba del clasificador sobre datos sintéticos

Reproducible con: `python ejemplos/demo_fase1.py` (fecha de esta corrida: 2026-09-01).

## Montaje

- **Calibración**: 60 mediciones sintéticas por polímero, ruido gaussiano isotrópico σ = 0,02
  en el plano de phasores. Centroides FLIM derivados de tiempos de vida *ilustrativos*
  (no medidos) con `phasorpy.lifetime.phasor_from_lifetime` a 80 MHz; centroides
  espectrales colocados a mano sobre el arco. Ver `tests/datos_sinteticos.py`.
- **Muestra**: 50 partículas por polímero (σ = 0,025, 25 % más ancho que la calibración,
  para simular condiciones de adquisición peores) + 80 partículas de "materia orgánica /
  autofluorescencia": nube ancha (σ = 0,10) desplazada +0,22 del conjunto de clusters.
  Verdad de terreno: `no_clasificable`.
- **Regla de rechazo**: `confianza = 0,99` → umbral `chi2.ppf(0.99, df)` (df = 2 para una
  modalidad, df = 4 para fusión).

## Tabla de resultados

| modalidad | estrategia | exactitud | F1 macro (con NC) | F1 macro (solo polímeros) | rechazo materia org. | falso "no clasif." |
|---|---|---|---|---|---|---|
| flim | centroide | 0,884 | 0,883 | 0,880 | 0,925 | 0,033 |
| flim | knn | 0,895 | 0,894 | 0,894 | 0,812 | 0,000 |
| flim | gmm | 0,871 | 0,869 | 0,865 | 0,912 | 0,037 |
| espectral | centroide | 0,934 | 0,942 | 0,956 | 0,900 | 0,053 |
| espectral | knn | 0,939 | 0,946 | 0,964 | 0,725 | 0,000 |
| espectral | gmm | 0,924 | 0,933 | 0,950 | 0,875 | 0,060 |
| **fusion** | **knn** | **0,987** | **0,988** | **0,990** | **0,975** | **0,007** |
| fusion | centroide | 0,900 | 0,914 | 0,932 | 1,000 | 0,127 |
| fusion | gmm | 0,905 | 0,919 | 0,936 | 1,000 | 0,120 |

`rechazo materia org.` = fracción de partículas orgánicas correctamente marcadas
`no_clasificable`. `falso "no clasif."` = fracción de polímero real marcado por error como
`no_clasificable`.

## Lecturas

1. **La fusión FLIM + espectral funciona.** `fusion + knn` alcanza exactitud 0,987 y
   F1 = 0,990 sobre polímeros, por encima de cualquier modalidad sola. Es la evidencia
   sintética de la tesis del proyecto (ningún antecedente combina ambas modalidades).
   Los números quedan en el rango de FIMAP (Ho et al. 2025: F1 = 94,7 %) y por encima de
   Meyers et al. (2022: 88,1 % en identificación de polímero).

2. **FLIM solo tiene pares solapados.** Con los lifetimes ilustrativos, PVC/PET y
   LDPE/HDPE quedan cerca en el plano de phasores y se confunden (exactitud ~0,88). La
   modalidad espectral los separa (~0,93) y la fusión los resuelve del todo. Esto hay que
   verificarlo con los lifetimes reales del equipo, pero es coherente con que Sancataldo
   et al. (2020) reporten solapamiento parcial entre algunos polímeros usando solo FLIM.

3. **El rechazo paramétrico (centroide/gmm) es sensible al desajuste de ruido
   calibración↔muestra.** Con la muestra 25 % más ancha que la calibración, `fusion +
   centroide` marca 12,7 % de polímero real como `no_clasificable` (df = 4 amplifica el
   efecto). El rechazo **no paramétrico de `knn`** —umbral aprendido del cuantil empírico
   de la calibración × 1,5— es mucho más robusto (0,7 % de falsos). **Recomendación para
   Fase 3**: usar `knn` para la decisión de rechazo, o estimar la covarianza de cluster
   con un factor de inflación, o fijar `confianza` por validación cruzada sobre la
   calibración real.

4. **El modo de falla es el deseado para muestras ambientales.** La precisión por
   polímero es 1,000 en todas las configuraciones: cuando el clasificador se equivoca,
   manda la partícula a `no_clasificable`, no a otro polímero. Para matrices con materia
   orgánica y autofluorescencia esto es lo correcto (mejor "no sé" que un falso positivo
   de plástico). El costo es la baja precisión de la clase `no_clasificable` (0,68 en
   `fusion+centroide`): absorbe tanto la materia orgánica real como el polímero mal medido.

## Artefactos generados

`ejemplos/salida_demo/`:

**Tablas** — `asignaciones.csv` (una fila por partícula, con `polimero_real`,
`polimero_predicho`, `score_rechazo`), `calibracion.csv`, `metricas_resumen.txt`,
`metricas_por_clase.csv`, `matriz_confusion.csv`.

**Figuras** (`figuras/`, PNG a 300 dpi + PDF vectorial para póster) —
generadas por `napari_mp_classifier.reportes`:

| archivo | qué muestra |
|---|---|
| `phasores_flim_knn.*` | diagrama de phasores FLIM: 6 clusters de referencia (centroide + elipse de covarianza 2σ) y partículas clasificadas; los aros rojos son errores. Se ven los pares que FLIM sola confunde (HDPE/LDPE/PP). |
| `phasores_espectral_knn.*` | ídem, modalidad espectral: separa los pares que FLIM confunde. |
| `phasores_fusion_knn.*` | fusión 4D en dos paneles (proyección FLIM y proyección espectral): prácticamente sin errores. |
| `matriz_confusion_fusion_knn.*` | mapa de calor normalizado por fila (recall por clase) del caso ganador. |
| `metricas_por_polimero_fusion_knn.*` | barras de precisión / recall / F1 por polímero. |
| `comparacion_modalidades.*` | exactitud y F1(polímeros) por modalidad × estrategia (las 9 combinaciones). |

Paleta categórica de los 6 polímeros en orden fijo y validada para daltonismo
(bandas de luminosidad, piso de croma, separación CVD entre pares adyacentes);
como el color solo no separa 6 clases con seguridad para todo tipo de daltonismo,
cada polímero lleva además un **marcador propio** y **etiqueta directa** sobre su
cluster (codificación secundaria: posición + forma + rótulo portan la identidad).

## Limitaciones de esta prueba

- Datos **sintéticos**: clusters gaussianos, sin estructura espacial ni artefactos de
  microscopía. No reemplaza la validación con `.sdt`/`.czi` reales.
- Los tiempos de vida y posiciones espectrales son **ilustrativos**, elegidos para dar 6
  clusters; los valores reales pueden estar más o menos separados.
- La "materia orgánica" es una sola nube gaussiana ancha; en la práctica es multimodal
  (celulosa, quitina, restos celulares, cada uno con su firma).
