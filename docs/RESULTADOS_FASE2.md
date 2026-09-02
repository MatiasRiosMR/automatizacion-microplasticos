# RESULTADOS_FASE2.md — segmentación + features + clasificación por ROI

Reproducible con: `python ejemplos/demo_fase2.py` (fecha de esta corrida: 2026-09-02).
Estadísticos agregados: media ± desvío sobre 10 imágenes sintéticas (semillas 0–9).

## Montaje

- **Imagen de muestra sintética** (`tests/datos_sinteticos.generar_imagen_muestra`):
  campo de 320×320 px con perfil de intensidad súper-gaussiano por partícula.
  - 4 partículas por polímero (24 en total), separación mínima 2,3× el radio máximo.
  - 4 partículas extra pegadas a otra (pares en contacto) → prueban `watershed`.
  - 8 cuerpos de "materia orgánica / autofluorescencia": ~1,6× más grandes, más tenues,
    firma de phasor difusa (ruido ×3) y desplazada +0,22 del conjunto de clusters.
  - Fondo gaussiano. Phasor por píxel = centroide del material + ruido; en los solapes,
    promedio ponderado por intensidad (genera dispersión intra-ROI en el borde).
  - Verdad de terreno: imagen de `labels` + polímero por label.
- **Segmentación** (`segmentacion.segmentar`): umbral Otsu o K-means (`[intensidad, g, s]`,
  enfoque FIMAP) → cierre morfológico + relleno de huecos + descarte de objetos < 8 px →
  `watershed` sobre la transformada de distancia (relieve = −distancia − intensidad
  normalizada) → reindexado 1..n.
- **Features** (`features.extraer_features`): por ROI — phasor FLIM y espectral (mediana
  espacial vía `phasorpy.phasor.phasor_center`), dispersión intra-ROI, intensidad
  total/media, área (px y µm²), excentricidad, solidez, extensión, relación de aspecto,
  perímetro, centroide.
- **Clasificación**: se reusa **tal cual** el clasificador de la Fase 1
  (`fusion` + `knn`, `confianza=0,99`) calibrado con datos sintéticos. La verdad de
  terreno por ROI se asigna por solape (IoU ≥ 0,3) con la segmentación verdadera.

## Segmentación

| método | IoU (ROIs emparejadas) | precisión detección | recall detección | nº ROIs |
|---|---|---|---|---|
| **umbral** (Otsu) | **0,813 ± 0,015** | 0,895 ± 0,047 | **0,769 ± 0,065** | ~30 |
| kmeans (FIMAP) | 0,769 ± 0,020 | **0,915 ± 0,057** | 0,686 ± 0,053 | ~26 |

- **IoU 0,81** con el umbralado, en el rango de FIMAP (Ho et al. 2025: IoU 87,7 %).
- El **K-means** que usa la firma de phasor sube la precisión de detección (descarta
  cuerpos difusos de materia orgánica que comparten brillo pero no firma) **a costa de
  recall**. El umbralado detecta más objetos —incluidos los orgánicos, que después
  rechaza el clasificador— y es el método por defecto de `segmentar`.
- El recall no llega a 1 sobre todo por pares en contacto muy solapados que `watershed`
  no separa y por cuerpos orgánicos tenues fragmentados.

## Clasificación de las ROIs

| métrica | valor |
|---|---|
| exactitud sobre ROIs de **polímero** bien segmentadas | **0,996 ± 0,012** |
| rechazo de materia orgánica (umbral) | 0,63 ± 0,30 |
| rechazo de materia orgánica (kmeans) | ~0 (casi no la segmenta) |

1. **El clasificador de la Fase 1 se traslada sin cambios a features de ROI reales.**
   Cuando una partícula de polímero se segmenta con IoU razonable, su phasor de ROI
   (mediana espacial) cae dentro del cluster correcto: exactitud ≈ 1,0 sobre polímeros.
   Es la señal de que el cuello de botella del pipeline es la **segmentación**, no la
   clasificación.

2. **El rechazo de materia orgánica a nivel de ROI es más ruidoso que en la Fase 1**
   (0,63 vs. 0,975). Motivo: la Fase 1 clasificaba puntos con el ruido de "partícula
   individual"; acá el phasor de ROI es la **mediana sobre cientos de píxeles**, con muy
   poca dispersión, así que el núcleo brillante de un cuerpo orgánico puede caer dentro
   del umbral χ²/knn de un cluster. La varianza alta (± 0,30) confirma que depende de
   dónde caiga esa nube.
   → **Implicancia para la Fase 3**: para matrices complejas (ambientales, fagocitos) hay
   que meter el paso de **`desmezcla.py`** (spectral unmixing con `phasorpy.component`,
   ver `docs/SPECTRAL_UNMIXING.md`) *antes* de clasificar, para separar la fracción
   NR-MP de la autofluorescencia; y/o subir `confianza`, y/o usar la dispersión intra-ROI
   como feature de rechazo adicional.

## Artefactos generados

`ejemplos/salida_demo_fase2/` (ignorado por git):

- `features_y_asignaciones.csv` — una fila por ROI: features de forma + phasor +
  `polimero_real`, `polimero_predicho`, `score_rechazo`.
- `metricas_resumen.txt`, `metricas_por_clase.csv`, `matriz_confusion.csv`.
- `figuras/segmentacion_umbral.*`, `figuras/segmentacion_kmeans.*` — imagen con las ROIs.
- `figuras/clasificacion_rois.*` — ROIs coloreadas por polímero predicho, rotuladas.
- `figuras/phasores_rois.*` — phasor de cada ROI sobre los clusters de referencia.
- `figuras/matriz_confusion_rois.*`.

## Limitaciones de esta prueba

- Imagen **sintética**: blobs súper-gaussianos, sin textura interna, sin artefactos de
  microscopía (viñeteo, PSF anisótropa, bleed-through espectral, fotoblanqueo).
- El phasor por píxel se genera desde el centroide del material; no proviene de una
  señal FLIM/espectral simulada realista. La validación con `.sdt`/`.czi` reales sigue
  pendiente (`docs/PREGUNTAS_DATOS.md`).
- La "materia orgánica" es una sola familia (una nube de phasor); en la práctica es
  multimodal (celulosa, quitina, restos celulares).
- No se probó todavía la vía de fagocitos (`restringir_a_mascara`): falta una máscara
  celular sintética realista, se agrega en la Fase 5 con las muestras de Mo/PMN.
