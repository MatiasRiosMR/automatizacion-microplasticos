# Fase 0 — Diseño y evaluación de dependencias

Fecha: 2026-09-01. Entorno de evaluación: Linux, Python 3.14.6 (conda), pip 26.1.

## 1. ¿`phasorpy` cubre la lectura de formatos y el cálculo de phasores?

**Sí, ampliamente.** Se instaló `phasorpy 0.12` en Python 3.14 sin problemas
(dependencias: numpy 2.5, scipy 1.18, xarray, tifffile, matplotlib, scikit-learn, pandas).

### Lectores de formato crudo (`phasorpy.io`)

Verificado por introspección del paquete instalado:

| Función | Formato | Modalidad | Ejes que devuelve |
|---|---|---|---|
| `signal_from_sdt()` | Becker & Hickl SDT | **FLIM** (TCSPC) | `QCYXH` |
| `signal_from_czi()` | Zeiss CZI | **hiperespectral** o RGB | dim `C` |
| `signal_from_lsm()` | Zeiss LSM | **hiperespectral** | coords de longitud de onda |
| `signal_from_ptu()` | PicoQuant PTU | FLIM | `TYXCH` |
| `signal_from_lif()` | Leica LIF | hiperespectral | coords de longitud de onda |
| `signal_from_fbd()` | FLIMbox FBD | FLIM | `TCYXH` |
| + `ometiff`, `imspector_tiff`, `flif`, `bh/bhz`, `flimlabs_json`, ... | | | |

**Conclusión:** los dos formatos que el equipo va a entregar (`.sdt` para FLIM, `.czi`
para espectral) están cubiertos nativamente. No hay que escribir lectores propios.

Todos los lectores devuelven un `xarray.DataArray` con:
- `data`: array NumPy
- `dims`: códigos de eje (p. ej. `YXH`)
- `coords`: p. ej. `coords['H']` = bins del histograma en ns; longitudes de onda para espectral
- `attrs`: metadatos (frecuencia en MHz, etc.)

### Cálculo de phasores (`phasorpy.phasor` y `phasorpy.lifetime`)

- `phasor_from_signal(signal, axis=..., harmonic=...)` — coordenadas (g, s) desde señal
  temporal, frecuencial o **hiperespectral** (mínimo 3 muestras equiespaciadas por el eje).
  Para datos hiperespectrales **no hace falta calibración** (la longitud de onda es absoluta).
- `phasor_calibrate(real, imag, ref_real, ref_imag, frequency, lifetime=...)` — calibra
  FLIM contra una imagen de referencia de lifetime conocido (necesario para `.sdt`).
- `phasor_center(real, imag, method='mean'|'median')` — centroide de un cluster → esto es
  exactamente lo que necesitamos para los 6 polímeros de referencia.
- `phasor_from_lifetime(frequency, lifetime, fraction=...)` — para generar clusters
  sintéticos realistas en Fase 1.
- `phasor_threshold(...)`, `phasor_filter_median/pawflim(...)` — limpieza previa.
- `phasor_nearest_neighbor(...)` — búsqueda de vecino más cercano con umbral de distancia
  opcional → línea base de clasificación ya provista.

### Cursores / máscaras (`phasorpy.cursor`)

- `mask_from_circular_cursor()`, `mask_from_elliptic_cursor()`, `mask_from_polar_cursor()`
  → generación de máscaras a partir de regiones en el plano de phasores (útil para definir
  cada cluster de referencia y para back-projection en napari).
- `pseudo_color()` → coloreado de la imagen según cluster asignado.

## 2. ¿`napari-phasors` como base, o paquete independiente?

`napari-phasors` (readthedocs: napari-phasors.readthedocs.io) está construido **sobre
`phasorpy`** y aporta: lectura de ~15 formatos, calibración con imagen de referencia,
cursores interactivos, tabla de estadísticas, exportación a OME-TIF/CSV, y **análisis
batch headless**.

### Decisión: **paquete independiente que depende de `phasorpy`.**

Justificación:

1. **Acople y control de versiones.** `napari-phasors` es un plugin de aplicación (UI),
   no una librería pensada para extenderse por herencia. Depender de él arrastra `napari`
   + Qt en todas las fases, incluso en la CLI y los tests que no necesitan GUI.
2. **Python 3.14.** `phasorpy` instala limpio; `napari` (Qt) es más frágil en 3.14 y
   añadiría fricción a las Fases 1-3, que son puro cálculo. La integración napari (Fase 4)
   se aísla en `napari_integracion/` y puede pedir un entorno 3.11/3.12 aparte.
3. **El valor propio está en la capa de clasificación**, no en re-graficar phasores.
   `phasorpy` ya da todo lo que `napari-phasors` usa por debajo (lectura + cálculo +
   cursores + centro de cluster). Construir sobre `phasorpy` directamente evita una capa
   intermedia que no aporta a lo que hay que construir.
4. **Interoperabilidad, no dependencia.** El formato de exportación de phasores de
   `napari-phasors` (CSV con columnas de coordenadas, OME-TIF) se adopta como formato de
   entrada aceptado en `calibracion.py`, para que quien ya use `napari-phasors` pueda
   alimentar este clasificador sin fricción. La integración de Fase 4 será un plugin napari
   propio y liviano que **puede convivir** con `napari-phasors` en el mismo visor.

### Lo que sí se reutiliza de `napari-phasors` conceptualmente

- Su convención de columnas de CSV de phasores (para `calibracion.cargar_phasores_csv`).
- La idea de "calibración con imagen de referencia de lifetime conocido" (se delega en
  `phasorpy.lifetime.phasor_calibrate`).

## 3. Stack de dependencias propuesto

| Paquete | Uso | Estado |
|---|---|---|
| `phasorpy>=0.12` | lectura de crudo + cálculo de phasores | instalado OK |
| `numpy`, `scipy` | base numérica | instalado (numpy 2.5) |
| `pandas>=2` | tablas de resultados y CSV | instalado (3.0.5) |
| `scikit-learn>=1.4` | KNN, GMM, métricas | instalado (1.9.0) |
| `scikit-image>=0.24` | segmentación (K-means vía `sklearn`, watershed, threshold) | instalado (0.26.0) |
| `tifffile` | E/S de TIFF | instalado |
| `matplotlib>=3.8` | diagramas de phasores y overlays | instalado (3.11.1) |
| `pytest` | tests | a instalar en `[dev]` |
| `napari` + `magicgui` + `qtpy` | solo Fase 4, extra opcional `[napari]` | **pendiente** (probable 3.11/3.12) |

`plotly` queda como opción futura para reportes HTML interactivos; la línea base usa
`matplotlib`.

## 4. Ajustes a la estructura de carpetas propuesta

Se mantiene la estructura del prompt con dos cambios menores:

- `src/napari_mp_classifier/` (layout `src/`, ya en el prompt) — OK.
- Se agrega `src/napari_mp_classifier/io_crudo.py`: wrapper fino sobre `phasorpy.io` que
  centraliza "de `.sdt`/`.czi` a coordenadas (g, s) por píxel", para no repetir la lógica
  de ejes/calibración en `calibracion.py`, `segmentacion.py` y `cli.py`.
- Se agrega `src/napari_mp_classifier/fusion.py`: combinación de las features de phasor
  FLIM + espectral en un único vector por ROI. Es el núcleo diferenciador del proyecto y
  merece módulo propio en vez de esconderse en `features.py`.
- `datos/` (git-ignorado) para las imágenes reales que entregue el equipo.

Estructura resultante:

```
src/napari_mp_classifier/
  __init__.py
  io_crudo.py        # .sdt/.czi -> (g,s) por píxel (wrapper de phasorpy.io)  [NUEVO]
  calibracion.py     # carga/cálculo de phasores de los 6 polímeros de referencia
  segmentacion.py    # detección de ROIs (K-means / watershed / threshold)
  features.py        # extracción de features por ROI
  fusion.py          # combinación FLIM + espectral por ROI  [NUEVO]
  clasificador.py    # centroide más cercano / KNN / GMM + "no clasificable"
  metricas.py        # precisión, exactitud, recall, F1, matriz de confusión
  reportes.py        # CSV, gráficos de phasores, resúmenes
  napari_integracion/
  cli.py
```

## 5. Preguntas abiertas para el equipo (bloquean el esquema de calibración)

Ver `docs/PREGUNTAS_DATOS.md`. Resumen:

1. Datos de calibración de los 6 polímeros: ¿`.sdt` + `.czi` crudos, o coordenadas de
   phasor ya calculadas (CSV/OME-TIF de `napari-phasors`)? ¿Uno por polímero o varios?
2. FLIM: frecuencia de modulación (MHz) y fluoróforo/lifetime de referencia para
   `phasor_calibrate`.
3. Espectral: rango de longitudes de onda y número de canales del λ-stack.
4. ¿Los `.sdt` y `.czi` de una misma partícula están registrados/alineados espacialmente,
   o son adquisiciones separadas? (define si la fusión es por píxel o por cluster).
5. Muestras: ¿tenemos imágenes con verdad de terreno (partículas de polímero conocido en
   matriz ambiental / con células) para medir las métricas?
6. Envejecimiento (Meyers 2024): ¿calibración solo con virgen, o hay material degradado?

## 6. Estado

- [x] `phasorpy` evaluado e instalado — cubre `.sdt` y `.czi` nativamente.
- [x] Decisión de arquitectura: paquete independiente sobre `phasorpy`.
- [x] Stack de dependencias definido (napari diferido a Fase 4).
- [ ] Confirmación del equipo sobre formato de datos reales (`docs/PREGUNTAS_DATOS.md`).
- [→] Se avanza en paralelo con Fase 1 (datos sintéticos) que no depende de esas respuestas.
