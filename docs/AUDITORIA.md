# AUDITORIA.md — revisión del código y de la validación (octubre 2026)

Revisión completa del paquete hecha el 2026-10-06, después de la primera prueba con datos
reales (`docs/RESULTADOS_PRUEBA_LAMAE.md`). Cada hallazgo se verificó corriendo el código
(no por lectura). Las correcciones se integraron el mismo día; este documento registra qué se
encontró, qué se corrigió y qué queda pendiente.

Los números de esta página se re-midieron el 2026-10-07 con el código ya corregido
(fusión FLIM + espectral, `confianza = 0,99`).

## Resumen

- **18 hallazgos**: 4 críticos, 6 altos, 5 medios, 3 bajos (16 del código y la validación +
  2 del FLIM real que aparecieron al calibrar con la IRF).
- **13 corregidos**, 3 parciales, 2 pendientes (ver tabla).
- Lo más importante que **no** se resuelve con código: en un escenario realista la
  autofluorescencia que cae cerca de un polímero no se separa solo con la distancia al
  cluster (AUROC del rechazo ≈ 0,93). Corregir el umbral cambia *dónde* cae el corte, no la
  capacidad de separar.

## Hallazgos

| # | Severidad | Área | Hallazgo | Estado |
|---|---|---|---|---|
| 1 | Crítico | Clasificador | Una ROI con phasor NaN se asignaba en silencio a HDPE con `centroide` (`argmin` sobre NaN → índice 0, y `NaN > 1` es falso → no se rechazaba); con `knn`/`gmm` el pipeline se caía con `ValueError`. Pasa en ROIs de pocos fotones. | ✔ NaN → `no_clasificable` con score `inf` en las tres estrategias |
| 2 | Crítico | Clasificador | El umbral χ² supone μ y Σ exactos. Con 10 mediciones por polímero en fusión 4D, `confianza = 0,99` rechazaba **23 %** de polímero real (nominal 1 %). | ✔ Región de predicción de **Hotelling** por clase (`umbral="hotelling"`, por defecto): 1,2 % en el mismo caso. Advierte si `n ≤ p + 1` |
| 3 | Crítico | Validación | El escenario sintético es trivialmente separable: covarianzas isotrópicas e idénticas, «materia orgánica» a +0,22 en todas las coordenadas, calibración y test del mismo generador. | ◐ Se mide y reporta un escenario realista (abajo); falta incorporarlo a `tests/datos_sinteticos.py` |
| 4 | Crítico | Datos reales | El `.sdt` de LAMAE tiene **dos detectores** y se leía el más débil (3 M vs. 63 M fotones). | ✔ `phasores_desde_sdt(detector=None)` elige el de más fotones; `listar_detectores_sdt` |
| 5 | Alto | Clasificador | GMM no supervisado: re-ajustaba EM sin etiquetas y mapeaba componentes al centroide más cercano (dos componentes podían caer en el mismo polímero). En pipeline/CLI/widget se entrenaba con datos sintéticos. | ✔ GMM supervisado: una gaussiana por polímero con sus propias mediciones |
| 6 | Alto | Clasificador | `centroide` asignaba por `argmin(d²)` sin el término `log|Σ|`: el cluster más ancho (p. ej. polímero envejecido) «roba» partículas. | ✔ Asignación **QDA** (`d² + log|Σ|`); el rechazo usa el d² de la clase asignada |
| 7 | Alto | Clasificador | KNN: voto sin ponderar, desempate **alfabético** (`np.unique`), umbral = cuantil × 1,5 ad hoc. Acepta el 79 % de la materia orgánica en el escenario realista. | ✔ Voto ponderado por 1/d, desempate por cercanía, `umbral_knn="conformal"` opcional. Estrategia por defecto pasa a `centroide` |
| 8 | Alto | CLI | `--modalidad espectral` fallaba siempre que el `.npz` traía los 4 canales (el pipeline re-deducía la modalidad). | ✔ Se filtran los canales antes de analizar |
| 9 | Alto | Métricas | La evaluación del pipeline sobreestimaba la exactitud: emparejamiento con IoU ≥ 0,3 no 1 a 1 (una ROI cubría dos verdaderas) y las no detectadas desaparecían. | ◐ Emparejamiento 1 a 1 (húngaro) en `emparejar_rois`; las falsas detecciones cuentan. Las no detectadas siguen fuera de `reporte_clasificacion` (están en `reporte_segmentacion`) |
| 10 | Alto | Datos reales | La corrección «pico → bin 0» del FLIM subestima el lifetime ~15 %. | ✔ Calibración con IRF del flanco, como SPCImage (`calibracion_flim.py`, `docs/CALIBRACION_FLIM.md`) |
| 11 | Medio | Datos reales | FLIM con muy pocos fotones (LAMAE: mediana 2, p99 78 por píxel). `phasor_filter_median` + `phasor_threshold` de phasorpy compactan la nube 3× (dispersión 0,068 → 0,023) y el pipeline no los usa. | ✘ Pendiente |
| 12 | Medio | IO | `PhasoresImagen.intensidad` es la **media** por bin (lo que devuelve `phasor_from_signal`), no la suma como decía el docstring. | ✔ Documentado; se agregó `fotones` (suma) |
| 13 | Medio | Calibración | No había ida y vuelta a disco: `guardar_csv` solo escribe centroides, sin covarianzas ni metadatos. Regularización fija 1e-6. | ◐ `guardar_json` / `cargar_json` (centroides, covarianzas, n, metadatos). Falta *shrinkage* (Ledoit-Wolf) para n chico |
| 14 | Medio | Features | Phasor por ROI = mediana marginal de g y s, no promedio ponderado por intensidad. Válido pero no configurable ni documentado. | ✘ Pendiente (`metodo_centro`) |
| 15 | Medio | Fusión | `fusionar_por_roi` emparejaba de forma voraz (dependía del orden de filas). | ✔ `linear_sum_assignment` sobre la matriz de distancias |
| 16 | Bajo | Figuras | Elipse 2σ cuando la frontera real es χ²₀,₉₉; colores de polímero reusados para métricas; sin intervalos, sin barra de escala, sin ticks de lifetime/λ; sin salida HTML. | ✔ en el informe nuevo (`informe_html.py`, `docs/INFORMES.md`); las figuras de `reportes.py` no se tocaron |
| 17 | Bajo | Docs | `separar_contacto`: docstring con valores por defecto distintos del código; en watershed la intensidad casi no pesa en el relieve. | ◐ Docstring alineado (3 / 0,4); normalización del relieve pendiente |
| 18 | Bajo | Métricas | F1 macro por defecto incluye `no_clasificable` como clase; sin exactitud balanceada ni AUROC del rechazo. | ✔ `exactitud_balanceada` e `incluir_no_clasificable`; AUROC se mide en la auditoría, no está en `metricas.py` |

✔ corregido · ◐ parcial · ✘ pendiente.

## Escenarios de validación

**Original** (el de `tests/datos_sinteticos.py`): calibración con 60 mediciones por
polímero (ruido 0,02), partículas de prueba con ruido 0,025, materia orgánica lejos de
todos los clusters. 10 semillas.

**Realista** (definido en la auditoría): covarianzas anisotrópicas y distintas por polímero,
**15 mediciones** por polímero, materia orgánica mitad como nube ancha entre clusters y
mitad como autofluorescencia compacta a ≈ 3σ de un polímero. 12 semillas.

Fusión FLIM + espectral, `confianza = 0,99`, código del repo al 2026-10-07 (media ± sd):

| Escenario | Estrategia | Exactitud balanceada | F1 polímeros | Polímero rechazado | Materia orgánica aceptada | AUROC rechazo |
|---|---|---|---|---|---|---|
| Original | `centroide` (QDA + Hotelling) | 0,943 ± 0,014 | 0,965 | 6,7 % | 0,3 % | 0,999 |
| Original | `knn` | 0,995 ± 0,004 | 0,996 | 0,3 % | 1,1 % | 0,999 |
| Realista | `centroide` (QDA + Hotelling) | **0,917 ± 0,028** | **0,934** | **1,3 %** | **51 %** | 0,926 |
| Realista | `knn` | 0,886 ± 0,020 | 0,907 | 0,1 % | 79 % | 0,946 |

Lectura:

- En el escenario original `centroide` rechaza 6,7 % de polímero porque las partículas de
  prueba tienen **más ruido que la calibración** (0,025 vs. 0,02). El umbral de Hotelling
  está calibrado para la dispersión de la calibración y lo detecta. Es el comportamiento
  correcto, y es un aviso para datos reales: si la muestra es más ruidosa que la
  calibración (menos fotones, otro grado de envejecimiento), sube el rechazo.
- En el escenario realista `centroide` es la mejor opción y por eso es el valor por defecto.
  `knn` con su umbral por defecto deja pasar casi toda la materia orgánica.
- Con el umbral viejo (χ², mal calibrado) el escenario realista rechazaba 11,8 % de
  polímero y aceptaba 21 % de orgánico. El AUROC era el mismo: rechazaba más orgánico solo
  porque estaba descalibrado.

### Fusión frente a cada modalidad

Exactitud balanceada, media sobre las mismas semillas:

| Escenario | Estrategia | FLIM | Espectral | Fusión |
|---|---|---|---|---|
| Original | `centroide` | 0,888 | 0,940 | 0,943 |
| Original | `knn` | 0,897 | 0,957 | **0,995** |
| Realista | `centroide` | 0,843 | 0,912 | **0,917** |
| Realista | `knn` | 0,798 | 0,872 | 0,886 |

La fusión nunca queda por debajo de una modalidad sola. Sin embargo, en el escenario
realista con la estrategia por defecto la ventaja sobre el espectral es **marginal**
(dentro del desvío entre semillas). En estos datos sintéticos el FLIM aporta poco por
encima del espectral. Si eso se confirma con datos reales, la tesis del póster («combinar
FLIM + espectral») necesita que el FLIM separe polímeros que el espectral confunde.

## Pendientes

1. **Separar la autofluorescencia cercana a un polímero.** Elegir `confianza` con datos de
   validación reales (controles negativos) y sumar evidencia además de la posición del
   phasor (dispersión intra-ROI, forma, intensidad, fracción de desmezcla).
2. Filtrado del phasor FLIM antes de extraer features (hallazgo 11).
3. Llevar el escenario realista a `tests/datos_sinteticos.py` y a los tests (hallazgo 3).
4. Contar las partículas no detectadas en una métrica end-to-end (hallazgo 9).
5. *Shrinkage* de covarianzas para calibraciones chicas (hallazgo 13).
6. `metodo_centro` configurable en `features.py` (hallazgo 14).
7. Validación cruzada *leave-one-particle-out* cuando haya calibración real.

## Reproducir

El banco de experimentos de la auditoría (`experimentos.py`) quedó fuera del repo. Las
cifras de la tabla se obtienen con `ClasificadorPhasor` sobre
`generar_calibracion("fusion", 60, 0.02, s)` + `generar_particulas("fusion", 50, 80, 0.025,
semilla=100 + s)` (original) y sobre el generador realista descrito arriba.
