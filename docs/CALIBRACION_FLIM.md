# CALIBRACION_FLIM.md — calibración del phasor FLIM con la IRF (SPCImage)

## El problema

El phasor FLIM medido no es el de la muestra: está **rotado y escalado por la IRF**
(función de respuesta instrumental: pulso láser + detector + electrónica). En el `.sdt` de
LAMAE el phasor crudo cae **fuera del semicírculo universal** (detector 1: g = 0,016,
s = 0,668), así que no se puede leer un lifetime de él.

Hay dos formas estándar de calibrar:

1. **Fluoróforo de referencia** de lifetime conocido (fluoresceína 4,0 ns, rodamina B
   1,68 ns…), medido en la misma sesión → `phasor_calibrate`. Es lo que pide
   `docs/PREGUNTAS_DATOS.md` (pregunta 5) y el equipo todavía no lo entregó.
2. **IRF**, medida (reflexión del láser en el cubreobjetos, SHG) o **sintética**, estimada
   del propio decaimiento. Es lo que hace **SPCImage** (Becker & Hickl), el software que usa
   el equipo.

`calibracion_flim.py` implementa la opción 2 sin reimplementar nada de phasorpy.

## Método

El phasor calibrado es el medido dividido por el phasor de la IRF (en notación compleja,
z = g + i·s):

```
z_calibrado = z_medido / z_IRF
```

Es exactamente `phasorpy.lifetime.phasor_calibrate` con una referencia de **lifetime 0**,
cuyo phasor teórico es (1, 0).

La **IRF sintética** sale del flanco de subida del decaimiento total (suma de los píxeles
más brillantes). La derivada de `IRF ∗ exp(−t/τ)` es `IRF − (IRF ∗ exp)/τ`, y en el flanco
domina el primer término. Por eso la derivada positiva del flanco, suavizada, aproxima la
IRF (`irf_desde_flanco`).

## Validación

Decaimientos simulados (`simular_decaimiento`) con los parámetros del `.sdt` de LAMAE:
59,96 MHz, 1024 canales y la misma ventana del TAC.

| Método | Error en τφ a 3 ns | Error entre 1 y 4 ns |
|---|---|---|
| «Pico → bin 0» (`corregir_desfase=True`, el método anterior) | −14 % | −14 a −16 % |
| IRF del flanco (SPCImage) | +3 % | +2 a +6 % |

Por encima de ~5 ns el error crece por los límites de la ventana del TAC, no por la
calibración: con la IRF verdadera pasa lo mismo. El rango de Nile Red (1–4 ns) queda
cubierto.

## Resultado en LAMAE (PE + Nile Red)

El `.sdt` tiene **dos detectores** (dos módulos TCSPC sobre el mismo campo):

| Detector | Fotones | τφ calibrado | τm calibrado |
|---|---|---|---|
| 0 | 3,1 M | 2,88 ns | 3,34 ns |
| 1 (por defecto: el de más fotones) | 63,3 M | **2,69 ns** | **2,95 ns** |

- El método anterior (pico → 0, detector 0) informaba τφ ≈ 2,3 ns.
- τφ < τm → la emisión no es monoexponencial.
- Los dos detectores dan lifetimes distintos. Lo más probable es que midan **bandas de
  emisión distintas**: hay que preguntarle al equipo qué filtro tiene cada uno.

Reproducir: `python ejemplos/demo_lamae.py` (necesita `ejemplo_lamae/`, no versionada).

## Uso

```python
from napari_mp_classifier.io_crudo import listar_detectores_sdt, phasores_desde_sdt

listar_detectores_sdt("muestra.sdt")             # fotones por detector
ph = phasores_desde_sdt("muestra.sdt", calibrar_irf=True)          # IRF sintética
ph = phasores_desde_sdt("muestra.sdt", irf=irf_medida)              # IRF medida
ph = phasores_desde_sdt("muestra.sdt", ruta_referencia="fluo.sdt",
                        referencia_lifetime_ns=4.0)                  # fluoróforo
g, s, intensidad = ph
```

Para calibrar `g`/`s` ya calculados: `calibracion_flim.calibrar_con_irf(g, s, irf,
frecuencia_mhz)`, que devuelve además la fase y la modulación de la IRF aplicadas.

## Limitaciones y pedido al equipo

- La IRF sintética es una aproximación. **Medir la IRF una vez por sesión** (reflexión del
  láser en el cubreobjetos) la elimina y es la opción más simple para el equipo.
- Confirmar el filtro de emisión de cada detector.
- La calibración corrige fase y modulación, no el bajo número de fotones por píxel. Para
  eso falta el filtrado del phasor (`docs/AUDITORIA.md`, hallazgo 11).
