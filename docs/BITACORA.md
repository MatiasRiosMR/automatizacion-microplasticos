# BITÁCORA de desarrollo

Registro cronológico de decisiones y avances. Entradas nuevas arriba.

---

## 2026-09-01 — Fase 0 + Fase 1

### Fase 0 — Diseño y evaluación de dependencias  ✔

- Se instaló y evaluó **`phasorpy 0.12`** en el entorno (Python 3.14, conda). Instala
  limpio con numpy 2.5 / scipy 1.18 / xarray / tifffile / scikit-learn / pandas.
- Verificado por introspección: `phasorpy.io` provee `signal_from_sdt` (FLIM, ejes
  `QCYXH`) y `signal_from_czi` (hiperespectral, dim `C`). **Los dos formatos del equipo
  están cubiertos nativamente**; no hay que escribir lectores.
- `phasorpy.phasor` / `phasorpy.lifetime` proveen `phasor_from_signal`, `phasor_calibrate`,
  `phasor_center`, `phasor_from_lifetime`, `phasor_nearest_neighbor`, y
  `phasorpy.cursor` provee máscaras circular/elíptica/polar + `pseudo_color`.
- **Decisión de arquitectura**: `napari_mp_classifier` se construye como **paquete
  independiente que depende de `phasorpy`**, NO como fork de `napari-phasors`.
  Justificación completa en `docs/FASE_0_EVALUACION.md` §2. En síntesis: `napari-phasors`
  es un plugin de UI, no una librería para extender; depender de él arrastra napari+Qt a
  toda fase; el valor propio está en la capa de clasificación, no en re-graficar phasores.
  La integración napari (Fase 4) será un plugin propio liviano que convive con
  `napari-phasors`.
- Se define el stack de dependencias (`pyproject.toml`). `napari` queda como extra
  opcional `[napari]`, diferido a Fase 4 (Qt es frágil en Python 3.14).
- Ajustes menores a la estructura: se agregan `io_crudo.py` (wrapper de lectura cruda) y
  `fusion.py` (combinación FLIM+espectral, el diferenciador del proyecto).
- Se dejaron **12 preguntas para el equipo** en `docs/PREGUNTAS_DATOS.md` (formato de
  calibración, parámetros FLIM/espectral, registro entre modalidades, verdad de terreno,
  envejecimiento). Bloquean el esquema definitivo de calibración pero **no** la Fase 1.

### Convenciones de git  ✔

- Conventional Commits (`feat`, `fix`, `docs`, `test`, `chore`).
- Historial a nombre del autor humano (`matiasr <matijr.mr@gmail.com>`), sin líneas de
  co-autoría automáticas.

### Fase 1 — Datos sintéticos + clasificador base  ✔

- `tests/datos_sinteticos.py`: 6 clusters separables (centroides FLIM derivados de
  lifetimes ilustrativos con `phasor_from_lifetime`; centroides espectrales colocados a
  mano) + población de "materia orgánica / autofluorescencia" para probar el rechazo.
  Soporta modalidades `flim`, `espectral`, `fusion` (4D).
- `calibracion.py`: `Calibracion` (centroide + covarianza por polímero), constructores
  desde DataFrame / CSV, serialización. Genérico en la dimensión (2D o 4D).
- `clasificador.py`: `ClasificadorPhasor` con 3 estrategias —`centroide` (Mahalanobis),
  `knn`, `gmm`— y regla de **"no clasificable"** por umbral de score. Documentado por qué
  y cuándo una partícula queda no clasificable.
- `metricas.py`: `evaluar_clasificacion` → exactitud, precisión/recall/F1 (macro y por
  clase), matriz de confusión, en formato comparable con Meyers 2022 / FIMAP 2025.
- `reportes.py`: CSV de asignaciones + volcado de métricas a disco + figuras de
  calidad publicación (ver más abajo).
- **Tests: 18/18 en verde** (`pytest`). Cubren: construcción de calibración, exactitud
  > 0.9 en polímeros conocidos para las 3 estrategias, rechazo > 0.8 de materia orgánica
  con < 0.1 de falsos "no clasificable", orientación de la matriz de confusión, exclusión
  opcional de `no_clasificable` de las métricas macro, y que la fusión 4D no empeora.

### Prueba end-to-end sobre datos sintéticos  ✔

- `ejemplos/demo_fase1.py`: corre las 3 estrategias × 3 modalidades y vuelca reporte a
  `ejemplos/salida_demo/`. Resultados en `docs/RESULTADOS_FASE1.md`.
- Al probar salió a la luz que un umbral fijo en σ **no** funciona entre estrategias ni
  dimensiones (KNN nunca rechazaba; la fusión 4D rechazaba de más). Se rediseñó la regla
  de "no clasificable":
  - Parámetro `confianza` (nivel de confianza, def. 0.99) en vez de `umbral_no_clasificable`.
  - centroide/gmm: umbral `chi2.ppf(confianza, df=n_features)` — consciente de la dimensión.
  - gmm ahora usa sus covarianzas ajustadas (antes era idéntico a centroide).
  - knn: umbral = cuantil `confianza` de las distancias intra-clase de la calibración × 1.5.
  - `score` devuelto = cociente score/umbral (>1 ⇒ rechazo), comparable entre estrategias.
- **Resultado clave**: `fusion + knn` → exactitud 0,987, F1 polímeros 0,990, rechazo de
  materia orgánica 0,975, falsos "no clasificable" 0,007. La fusión supera a cualquier
  modalidad sola → evidencia sintética de la tesis del proyecto.
- **Hallazgo para Fase 3**: el rechazo paramétrico (centroide/gmm) es sensible al
  desajuste de ruido calibración↔muestra; el de knn es robusto. Ver RESULTADOS_FASE1 §3.
- 27 tests en verde.

### Figuras de resultados — calidad publicación/póster

- `reportes.py` gana el módulo de figuras que anticipaba su docstring:
  `figura_phasores` (clusters de referencia con elipse de covarianza + partículas
  clasificadas; 1 panel por modalidad simple, 2 para la fusión 4D),
  `figura_matriz_confusion` (mapa de calor normalizado por fila),
  `figura_metricas_por_clase` (barras precisión/recall/F1) y `figura_comparacion`
  (modalidades × estrategias). `guardar_figura` escribe PNG 300 dpi + PDF vectorial.
- Paleta de los 6 polímeros en **orden fijo y validada para daltonismo** (skill
  `dataviz`: bandas de luminosidad, piso de croma, ΔE CVD entre pares adyacentes).
  Como el color solo no separa 6 clases para todo tipo de daltonismo, se agrega
  **codificación secundaria**: marcador propio por polímero + etiqueta directa
  sobre el cluster + elipse. Estilo sobrio compartido (`ESTILO_PUBLICACION`).
- `demo_fase1.py` ahora genera todas las figuras en `ejemplos/salida_demo/figuras/`
  y usa `fusion + knn` (el ganador) como caso detallado. Las figuras `phasores_*`
  muestran visualmente la tesis: FLIM sola confunde HDPE/LDPE/PP (aros rojos),
  la espectral los separa, la fusión los resuelve.
- +6 tests (`test_reportes.py`), 33 en verde. `conftest.py` fuerza backend Agg y
  cierra figuras tras cada test.

### Relevamiento: spectral unmixing de λ-stacks (a pedido del equipo)

- El equipo reporta que el linear unmixing de Fiji anda mal. Se relevó el ecosistema
  napari/Python → `docs/SPECTRAL_UNMIXING.md`.
- **Conclusión**: `phasorpy.component` (ya es dependencia) hace unmixing basado en
  phasores, model-free, con `phasor_component_fit` (N componentes, multi-armónico, admite
  1 componente desconocido para autofluorescencia), `phasor_component_fraction` (2),
  `phasor_component_graphical`, `phasor_component_mvc`. `napari-phasors` lo expone como
  "Component analysis". Es la vía recomendada y comparte espacio con la clasificación.
- Alternativas: napari-PICASSO (spillover ciego entre canales, GPU; no endmember de
  λ-stack), napari-hsi-analysis (exploración/UMAP, no unmixing), napari-musa (a revisar).
- Se planifica módulo `desmezcla.py` (Fase 3, adelantable a Fase 2 para muestras de
  fagocitos): separar fracción NR-MP de autofluorescencia antes de clasificar.

## 2026-09-02 — Reescritura de historia + Fase 2

### Limpieza de historia de git

- Se sacaron de **todos** los commits (con `git filter-repo`): `CLAUDE.md`,
  `prompt_claude_code_plan_de_trabajo.md`, `.claude/settings.json`. Siguen en el disco
  (untracked, ignorados). Force-push a `origin/master` (`59e7a73` → `232fad0`). Backup de
  la historia previa en un bundle local.
- Quien tenga un clon debe `git fetch && git reset --hard origin/master` o re-clonar.

### Fase 2 — segmentación + features  ✔

- **`segmentacion.py`**: `segmentar_umbral` (Otsu/Li/fijo), `segmentar_kmeans`
  (`[intensidad, g, s]` por píxel, enfoque FIMAP), `separar_contacto` (watershed sobre la
  transformada de distancia), `segmentar` (orquestador: umbral|kmeans → limpieza → watershed
  → filtro de tamaño), `restringir_a_mascara` (ROIs dentro de la máscara celular, para
  fagocitos — Park et al. 2020). Default `metodo="umbral"` (mejor recall/IoU sobre sintético).
- **`features.py`**: `extraer_features` → DataFrame por ROI con phasor FLIM/espectral
  (mediana espacial vía `phasorpy.phasor.phasor_center`), dispersión intra-ROI, intensidad,
  área (px/µm²), excentricidad, solidez, extensión, relación de aspecto, perímetro,
  centroide. `matriz_features` arma la `X` en el orden del clasificador.
- **`metricas.py`**: `evaluar_segmentacion` (IoU medio, precisión/recall de detección),
  `emparejar_rois` (IoU máximo verdad↔predicho).
- **`reportes.py`**: `figura_segmentacion` (imagen de intensidad + ROIs, contorno o
  coloreadas por polímero predicho).
- **`tests/datos_sinteticos.py`**: `generar_imagen_muestra` — campo 320×320 con blobs
  súper-gaussianos (6 polímeros + materia orgánica + pares en contacto) + phasor por píxel
  + verdad de terreno (labels + polímero por label).
- `ejemplos/demo_fase2.py` + `docs/RESULTADOS_FASE2.md`. **+21 tests (54 en verde).**
  `ruff` limpio en todo el repo (se aplicaron sus fixes también a archivos previos: orden
  de imports, comillas en anotaciones de tipo con `from __future__ import annotations`).

### Resultados clave (10 imágenes sintéticas)

- Segmentación (umbral): **IoU 0,81 ± 0,02** (nivel FIMAP: 0,877), precisión detección
  0,90, recall 0,77. K-means: IoU 0,77, precisión 0,92, recall 0,69 (usa la firma de
  phasor → más precisión, menos recall).
- **El clasificador de la Fase 1 se traslada sin cambios**: exactitud **0,996 ± 0,012**
  sobre ROIs de polímero bien segmentadas → el cuello de botella es la segmentación.
- **Hallazgo**: el rechazo de materia orgánica a nivel de ROI es más ruidoso que en la
  Fase 1 (0,63 ± 0,30 vs. 0,975), porque el phasor de ROI es la mediana de cientos de
  píxeles (casi sin dispersión) y el núcleo brillante de un cuerpo orgánico puede caer
  dentro del umbral. → En la Fase 3 hace falta `desmezcla.py` (`phasorpy.component`) antes
  de clasificar para matrices complejas, y/o usar la dispersión intra-ROI como feature.

### Fase 3 — pipeline + fusión + desmezcla + CLI  ✔

- **`pipeline.py`**: `analizar_muestra(canales, calibracion, …)` → `ResultadoMuestra`
  (features + `polimero_predicho` + `score_rechazo` + reportes de seg/clf si hay verdad).
  Deduce la modalidad de los canales presentes. Soporta `mascara_celular` (fagocitos).
- **`fusion.py`**: `fusionar_por_roi` (empareja ROIs de dos segmentaciones registradas por
  cercanía de centroide), `fusionar_por_decision` (combina dos clasificaciones
  independientes; acuerdo → polímero, desacuerdo/rechazo → `no_clasificable`).
- **`desmezcla.py`**: `fracciones_dos_componentes` / `fracciones_multi_componente`
  (envuelven `phasorpy.component`), `enmascarar_por_fraccion`, `phasor_mp_de_calibracion`.
- **`reportes.generar_reporte`**: informe unificado (CSV + métricas + figuras +
  `resumen_muestra.md`). `reportes.resumen_muestra`.
- **`cli.py classify`** funcionando sobre `.npz` (canales) + CSV de calibración → informe.
- `ejemplos/demo_fase3.py` + `docs/RESULTADOS_FASE3.md`. **78 tests en verde** (+24). ruff limpio.

### Resultados (12 imágenes sintéticas)

- Pipeline (fusión 4D + knn): IoU seg 0,81, exactitud clf 0,95, **exactitud sobre ROIs de
  polímero bien segmentadas 0,994 ± 0,014**. No degrada respecto a correr las etapas a mano.
- Fusión 4D ≈ fusión por decisión ≈ una modalidad sola sobre ROIs sintéticas limpias
  (el phasor de ROI es mediana de cientos de píxeles → ruido ~0). La ventaja de la fusión
  se ve bajo ruido/degradación (Fase 1: 0,987 vs 0,88–0,94; se reconfirma en Fase 5).
- Desmezcla separa las poblaciones: fracción NR-MP media **0,60 en polímero vs 0,13 en
  materia orgánica** → `enmascarar_por_fraccion` antes de segmentar debería estabilizar el
  rechazo de materia orgánica que hoy es ruidoso (0,65 ± 0,28).

### Fase 4 — plugin de napari  ✔

- Entorno dedicado **`napari-mp-env`** (conda, py 3.12, napari 0.9, phasorpy 0.12, PyQt6):
  los envs con napari que ya había (`napari-flim` py 3.10, `mnp-phasor-env` phasorpy 0.4)
  son incompatibles con `phasorpy >= 0.12` (que exige py ≥ 3.12).
- `napari_integracion/`: `napari.yaml` (manifiesto npe2, 2 widgets), `_widget.py`
  (`WidgetClasificador` → corre `analizar_muestra`, agrega capa `Labels` "clasificación MP"
  coloreada por polímero con `DirectLabelColormap`; features y `ResultadoMuestra` en la
  capa), `_phasor_plot.py` (`PhasorPlotWidget`: clusters de referencia + punto por ROI,
  **back-projection bidireccional** capa↔plot, selector de plano FLIM/espectral).
- `pyproject.toml`: entry-point `napari.manifest` + `pytest-qt` en `[dev]`.
- `tests/test_napari_integracion.py` (6 tests, headless con `QT_QPA_PLATFORM=offscreen`,
  se saltan si no hay napari). `ejemplos/demo_fase4.py`. `docs/RESULTADOS_FASE4.md`.
- **84 tests en verde en `napari-mp-env`** (78 base + 6 napari); 78 + 1 skip en la base.
- Limitación: las capturas del lienzo de napari necesitan display real (OpenGL); en
  headless se omiten. La lógica queda cubierta por los tests.

### Fase 5 — robustez, fagocitos, decisión de calibración, notebook  ✔

- **Decisión de calibración** (`docs/DECISION_CALIBRACION.md`): el equipo **no usa polímero
  virgen**. La calibración se hace sobre polímero envejecido con el estándar (abrasión +
  H2O2 [+ UV 1 h]), alineando calibración y muestra ambiental → sortea el modo de falla de
  Meyers 2024. Riesgo residual: variabilidad del grado de envejecimiento.
- `tests/datos_sinteticos.py`: `generar_particulas`/`generar_imagen_muestra` ganan
  `grado_envejecimiento` (modelo: los 6 clusters convergen a una firma común + inflación de
  ruido); `generar_mascara_celular` (máscara celular sintética para fagocitos).
- `ejemplos/demo_fase5.py` (6 experimentos) + `docs/RESULTADOS_FASE5.md`.
- `ejemplos/notebook_demo.ipynb` — recorrido end-to-end en Jupyter (Fases 1–5).
- `tests/test_robustez.py` (+11). **89 tests en verde** (base) / 95 en `napari-mp-env`.
- Respondida la pregunta 12 de `docs/PREGUNTAS_DATOS.md`.

### Resultados clave (sintético)

- **Ventana de tolerancia al desajuste de envejecimiento ~±0,15** (exactitud > 0,93),
  degradación gradual, modo de falla conservador (a `no_clasificable`, no a otro polímero).
- **`confianza = 0,995` recomendado** para muestras ambientales (recupera polímero real
  perdido sin bajar casi el rechazo de materia orgánica).
- **La fusión FLIM+espectral es más robusta al envejecimiento que cualquier modalidad
  sola en todo el rango** — la tesis del proyecto se sostiene bajo desajuste de dominio.
- Desmezcla previa: elimina 21 de 22 ROIs de materia orgánica (a costa de ~10 ROIs de
  polímero tenue). `restringir_a_mascara` aísla el MP fagocitado.

### Estado del proyecto

Fases 0–5 completas sobre datos sintéticos. **Falta la validación con `.sdt`/`.czi`
reales** (`io_crudo.py` + calibración medida + muestras ambientales / de cultivos), que
depende de que el equipo entregue los datos y responda `docs/PREGUNTAS_DATOS.md`.
