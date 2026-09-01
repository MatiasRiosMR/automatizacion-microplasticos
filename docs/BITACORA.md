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

- `.claude/settings.json` con `attribution.commit=""` y `attribution.pr=""`.
- `CLAUDE.md` con la instrucción explícita de no agregar `Co-Authored-By` / `Generated
  with Claude Code`.
- Pendiente: commit de prueba + `git log -1` para verificar (se hace al cerrar esta tanda).

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
- `reportes.py` (parcial): CSV de asignaciones + volcado de métricas a disco.
- **Tests: 18/18 en verde** (`pytest`). Cubren: construcción de calibración, exactitud
  > 0.9 en polímeros conocidos para las 3 estrategias, rechazo > 0.8 de materia orgánica
  con < 0.1 de falsos "no clasificable", orientación de la matriz de confusión, exclusión
  opcional de `no_clasificable` de las métricas macro, y que la fusión 4D no empeora.

### Próximo

- Confirmar con el equipo las respuestas de `docs/PREGUNTAS_DATOS.md`.
- Recibir `.sdt` y `.czi` de ejemplo → implementar `io_crudo.py` y validar la calibración
  real contra la sintética.
- Fase 2: segmentación (`segmentacion.py` + `features.py`).
