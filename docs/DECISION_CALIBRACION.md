# DECISION_CALIBRACION.md — calibración frente al envejecimiento

## Decisión

**La calibración de los 6 polímeros de referencia se hace sobre polímero envejecido de
forma controlada, no sobre polímero virgen.**

Protocolo (definido por el equipo): antes de teñir con Nile Red, a cada polímero de
referencia se le aplica

- **abrasión mecánica**,
- **peróxido de hidrógeno (H₂O₂)** — envejecimiento oxidativo y térmico,
- opcionalmente **UV, 1 h** — fotoenvejecimiento.

Los 6 clusters del plano de phasores (`calibracion.py`) quedan así definidos por la firma
de Nile Red sobre matriz **degradada**, no virgen.

## Por qué

Meyers et al. (2024, *Environ. Sci. Pollut. Res.*, https://doi.org/10.1007/s11356-024-35289-0)
mostraron que un clasificador calibrado **solo con polímero virgen** pierde fiabilidad
sobre microplástico ambiental, que llega meteorizado. La causa es un desajuste sistemático
de dominio: la muestra vive en una región del espacio de features que la calibración nunca
vio.

Calibrar con polímero envejecido con el **mismo tipo de degradación** que sufre el MP
ambiental (mecánica + oxidativa + fotoquímica) elimina ese desajuste sistemático: la
referencia y la muestra están en el mismo estado. Es la contramedida directa al hallazgo
de Meyers 2024.

## Riesgo residual y cómo se maneja

Queda una fuente de variabilidad: el MP ambiental tiene un **espectro** de grados de
meteorización, mientras que el estándar usa un protocolo fijo. Una partícula mucho más (o
mucho menos) degradada que el estándar puede caer desplazada de su cluster.

Cómo lo absorbe el pipeline (ver `docs/RESULTADOS_FASE5.md`, experimento 1):

1. **Ventana de tolerancia.** En sintético, con la calibración en el estándar, la exactitud
   se mantiene > 0,93 mientras el desajuste de envejecimiento de la muestra esté dentro de
   ~±0,15 (en unidades de la distancia cluster→firma-común). Fuera de eso degrada de forma
   **gradual**, no abrupta.

2. **Modo de falla conservador.** Cuando una partícula degradada se sale de su cluster, la
   regla `no_clasificable` la marca como no clasificable en vez de asignarla a otro
   polímero. En el barrido, la precisión por polímero se mantiene ~1,0: los errores son
   "no sé", no falsos positivos. Para matrices ambientales esto es lo correcto.

3. **`confianza` como perilla.** Bajo desajuste moderado (0,12), subir `confianza` de 0,99
   a **0,995–0,999** recupera el polímero real que se estaba perdiendo (fracción perdida
   3,5 % → 1,3 %) manteniendo el rechazo de materia orgánica en ~0,94. Es el punto de
   operación recomendado para muestras ambientales.

4. **Fusión FLIM + espectral.** La fusión es más robusta al desajuste de envejecimiento que
   cualquier modalidad sola en todo el rango probado (p. ej. a desajuste 0,15: fusión 0,93
   vs. espectral 0,89 vs. FLIM 0,79).

## Recomendaciones operativas

- Calibrar con **≥ 2 lotes** del estándar de envejecimiento por polímero, para que la
  covarianza del cluster capture la variabilidad del propio protocolo.
- Si se puede, sumar al set de calibración algunas mediciones con **grados de
  envejecimiento distintos** (p. ej. con y sin UV, distintos tiempos de H₂O₂): amplía la
  covarianza del cluster en la dirección de la deriva y hace la ventana de tolerancia más
  ancha. La calibración 4D ya admite mezclar lotes sin cambios de código.
- Fijar `confianza` por validación cruzada sobre la calibración real (no dejar el 0,99 por
  defecto a ciegas).
- Documentar en cada informe el protocolo de envejecimiento del estándar usado.

## Estado

Decisión tomada. Falta validarla con los `.sdt`/`.czi` reales del estándar envejecido y de
muestras ambientales (`docs/PREGUNTAS_DATOS.md`).
