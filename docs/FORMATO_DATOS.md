# FORMATO_DATOS.md — formatos de entrada y salida

## Entrada: calibración

### Opción A — CSV de coordenadas de phasor (recomendado para empezar)

Una fila por medición de polímero conocido. Columnas mínimas:

| Columna | Descripción |
|---|---|
| `polimero` | Código SPI: `PET`, `HDPE`, `PVC`, `LDPE`, `PP`, `PS` |
| `g_flim`, `s_flim` | Coordenadas de phasor FLIM (si hay modalidad temporal) |
| `g_esp`, `s_esp` | Coordenadas de phasor espectral (si hay modalidad λ-stack) |

Se aceptan solo FLIM, solo espectral, o las 4 columnas (fusión). Mínimo **2 filas por
polímero** (para estimar la covarianza del cluster). Los nombres de columna se pueden
cambiar al cargar (`Calibracion.cargar_phasores_csv(..., columnas=..., columna_etiqueta=...)`),
por lo que un CSV exportado de `napari-phasors` sirve directamente.

```python
from napari_mp_classifier import Calibracion
cal = Calibracion.cargar_phasores_csv(
    "datos/calibracion/phasores.csv",
    columnas=["g_flim", "s_flim", "g_esp", "s_esp"],
    columna_etiqueta="polimero",
)
```

### Opción B — imágenes crudas

- **FLIM**: `.sdt` (Becker & Hickl). `phasorpy.io.signal_from_sdt` → ejes `QCYXH`.
  Requiere frecuencia de modulación (MHz) y una imagen de referencia de lifetime conocido
  para `phasorpy.lifetime.phasor_calibrate`.
- **Espectral**: `.czi` (Zeiss). `phasorpy.io.signal_from_czi` → dimensión `C`
  (longitud de onda). Sin calibración (la longitud de onda es absoluta). Necesita ≥ 3
  canales equiespaciados.

El wrapper `io_crudo.py` unifica ambas rutas a `(g, s, intensidad)` 2D. **Pendiente**
hasta que el equipo entregue archivos de ejemplo — ver `docs/PREGUNTAS_DATOS.md`.

## Entrada: muestra a clasificar

Imagen de microscopía de Nile Red de una matriz compleja (ambiental o cultivo celular),
en `.tif`, `.sdt` o `.czi`. La segmentación (Fase 2) produce las ROIs.

## Salida

| Archivo | Contenido |
|---|---|
| `asignaciones.csv` | una fila por partícula: `id`, coordenadas de phasor, `polimero_predicho`, `score_rechazo` |
| `metricas_resumen.txt` | exactitud, precisión, recall, F1 (macro) + tablas |
| `metricas_por_clase.csv` | precisión / recall / F1 / soporte por polímero |
| `matriz_confusion.csv` | fila = verdad de terreno, columna = predicción |
| `phasores.png` | diagrama de phasores: clusters de referencia + partículas (Fase 3) |
| `overlay.png` | imagen de la muestra con ROIs delineadas y etiquetadas (Fase 3) |

`score_rechazo`: distancia al cluster más cercano (Mahalanobis en σ para
`centroide`/`gmm`; distancia euclídea media a los k vecinos para `knn`). Valores altos →
partícula lejos de todo cluster de polímero conocido.
