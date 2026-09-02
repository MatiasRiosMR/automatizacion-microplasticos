# RESULTADOS_FASE5.md — robustez y flujo de fagocitos

Reproducible con: `python ejemplos/demo_fase5.py` (fecha: 2026-09-02). Todo sobre datos
sintéticos; la validación con muestras reales queda pendiente de los `.sdt`/`.czi` del
equipo (`docs/PREGUNTAS_DATOS.md`).

La calibración representa el **estándar de envejecimiento** (abrasión + H₂O₂ [+ UV], ver
`docs/DECISION_CALIBRACION.md`). `grado_envejecimiento` mide cuánto se aparta el estado de
la muestra de ese estándar, en unidades de la distancia cluster→firma-común.

## 1. Desajuste de envejecimiento (calibración fusión + knn)

| grado | exactitud (polímeros) | polímero perdido → `no_clasificable` | rechazo mat. orgánica |
|---:|---:|---:|---:|
| −0,20 | 0,819 | 0,163 | 1,000 |
| −0,10 | 0,979 | 0,019 | 1,000 |
| **0,00** | **1,000** | **0,000** | **1,000** |
| 0,05 | 0,996 | 0,004 | 1,000 |
| 0,10 | 0,981 | 0,010 | 1,000 |
| 0,15 | 0,931 | 0,050 | 1,000 |
| 0,20 | 0,842 | 0,123 | 1,000 |
| 0,30 | 0,515 | 0,354 | 1,000 |

- **Ventana de tolerancia ~±0,15**: exactitud > 0,93. Degradación **gradual**, no abrupta.
- **Modo de falla conservador**: lo que se pierde va a `no_clasificable`, no a otro
  polímero (la precisión por polímero se mantiene ~1,0). El rechazo de materia orgánica no
  se ve afectado por la deriva (siempre 1,0).
- Figura: `ejemplos/salida_demo_fase5/robustez_envejecimiento.png`.

## 2. Ruido de adquisición

| σ del phasor de la muestra | exactitud | polímero perdido |
|---:|---:|---:|
| 0,015 | 1,000 | 0,000 |
| 0,025 (≈ calibración) | 0,998 | 0,002 |
| 0,040 | 0,817 | 0,179 |
| 0,060 | 0,408 | 0,575 |
| 0,090 | 0,127 | 0,850 |

Robusto mientras el ruido de la muestra no supere mucho al de la calibración (σ ≈ 0,02).
Más allá, de nuevo el modo de falla es "no clasificable", no misclasificación.

## 3. Punto de operación de `confianza` (desajuste de envejecimiento = 0,12)

| confianza | F1 (todas las clases) | polímero perdido | rechazo mat. orgánica |
|---:|---:|---:|---:|
| 0,90 | 0,855 | 0,192 | 0,988 |
| 0,95 | 0,908 | 0,110 | 0,988 |
| 0,975 | 0,948 | 0,052 | 0,975 |
| 0,99 (defecto) | 0,960 | 0,035 | 0,963 |
| **0,995** | **0,970** | **0,017** | 0,938 |
| 0,999 | 0,973 | 0,013 | 0,938 |

**Recomendación para muestras ambientales: `confianza = 0,995`** — recupera casi todo el
polímero real que el 0,99 perdía, a costa de bajar el rechazo de materia orgánica solo de
0,96 a 0,94. Fijarlo por validación cruzada sobre la calibración real.

## 4. Fusión vs. una sola modalidad, bajo desajuste de envejecimiento

| grado | FLIM sola | espectral sola | **fusión** |
|---:|---:|---:|---:|
| 0,00 | 0,887 | 0,994 | **1,000** |
| 0,05 | 0,873 | 0,988 | **1,000** |
| 0,10 | 0,844 | 0,946 | **0,975** |
| 0,15 | 0,785 | 0,887 | **0,929** |
| 0,20 | 0,738 | 0,775 | **0,819** |

**La fusión FLIM + espectral es más robusta al envejecimiento que cualquier modalidad
sola en todo el rango.** La tesis del proyecto no solo se sostiene en datos limpios
(Fase 1) sino también bajo el desajuste de dominio que es la preocupación de Meyers 2024.

## 5. Desmezcla integrada al pipeline (muestra con mucha materia orgánica)

| | ROIs totales | de materia orgánica | exactitud polímero | falsos "MP" de mat. orgánica |
|---|---:|---:|---:|---:|
| sin desmezcla | 49 | 22 | 1,000 | 4 |
| **con desmezcla** (`enmascarar_por_fraccion`, umbral 0,4) | 19 | 1 | 1,000 | 1 |

Enmascarar por fracción de Nile Red-MP **antes** de segmentar elimina 21 de las 22 ROIs de
materia orgánica y baja los falsos positivos de 4 a 1. Costo: también se pierden ~10 ROIs
de polímero real (49→19 en total), sobre todo las más tenues. Es un intercambio a calibrar
con el umbral según lo crítico que sea el falso positivo en cada tipo de muestra.

## 6. Flujo de fagocitos (`restringir_a_mascara`, Park et al. 2020)

Con una máscara celular sintética que cubre el 16 % del campo y "fagocita" el 60 % de las
partículas: de 29 ROIs segmentadas, quedan 17 dentro de células. `restringir_a_mascara`
aísla la señal de NR-MP fagocitado del resto del campo (autofluorescencia, MP libre) para
las muestras de monocitos / neutrófilos.

## Limitaciones

- Datos **sintéticos**: ver `docs/RESULTADOS_FASE2.md`. El modelo de envejecimiento
  (convergencia de los clusters hacia una firma común + inflación de ruido) es una
  hipótesis razonable pero no medida; hay que validarla con el estándar real.
- La máscara celular sintética son discos; las células reales tienen forma irregular y
  autofluorescencia estructurada.
- El componente de autofluorescencia de la desmezcla sigue siendo sintético.
