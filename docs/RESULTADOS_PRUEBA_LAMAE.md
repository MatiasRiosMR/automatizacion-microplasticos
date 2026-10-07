# RESULTADOS_PRUEBA_LAMAE.md — primera prueba con imágenes reales

Primera corrida de la tubería sobre datos reales del equipo (carpeta `ejemplo_lamae/`,
no versionada). Habilita `io_crudo.py`, que estaba pendiente de estos archivos.

## Datos

Una única partícula de **polietileno (PE)** teñida con Nile Red, objetivo 63x/1.4 óleo,
excitación 440 nm. Tres archivos, **tres adquisiciones distintas del mismo campo**:

| Archivo | Modalidad | Forma | Detalle |
|---|---|---|---|
| `PE-NileRed-63x-440.sdt` | FLIM (TCSPC) | 512×512×1024 | 59,96 MHz (del encabezado); 1024 bins sobre un período (16,7 ns) |
| `PE-NileRed-63x-lmada-440.czi` | λ-stack espectral | 28×440×440 | 451–721 nm, paso 10 nm; píxel 76,7 nm |
| `PE-NileRed-63x-zstack-440.czi` | z-stack | — | no espectral: no produce phasor |

FLIM y espectral **no están registrados** (512² vs 440², distinto píxel): la fusión con
estos archivos sería por cluster, no por píxel/partícula.

## Qué se implementó

`io_crudo.py` (antes stub que tiraba `NotImplementedError`):

- `phasores_desde_sdt(ruta, *, frecuencia_mhz=None, armonico=1, detector=None,
  calibrar_irf=False, irf=None, ruta_referencia=None, referencia_lifetime_ns=None,
  corregir_desfase=False)` → `PhasoresImagen` (FLIM). `listar_detectores_sdt(ruta)`.
- `phasores_desde_czi(ruta, *, eje_espectral="C", armonico=1)` → `PhasoresImagen` (espectral).
- `PhasoresImagen`: `g`, `s`, `intensidad` (2D) + metadata (`frecuencia_mhz`, `armonico`,
  `calibrado`, `modalidad`, `metadatos`, `fotones`). Se desempaqueta como
  `g, s, intensidad = ...`. `intensidad` es la media por bin/canal; `fotones`, la suma.

Reproducir: `python ejemplos/demo_lamae.py` → `ejemplos/salida_demo_lamae/`.
Informe de caracterización (HTML + PDF): `python ejemplos/informe_lamae.py`
(ver `docs/INFORMES.md`).

## Resultado

### Espectral — OK

- Phasor de la partícula (píxeles brillantes, armónico 1): **g = −0,616, s = 0,196**
  (ángulo 162,4°). Cluster compacto y unimodal (ver `phasor_espectral.png`).
- Centro espectral aparente ≈ **578 nm**: Nile Red en entorno apolar/hidrofóbico,
  consistente con una poliolefina como PE.
- Segmentación por umbral (Otsu, sin watershed): 1 ROI, **215 µm²** (≈ 16 µm de diámetro
  equivalente). Dispersión intra-ROI del phasor 0,046 → material homogéneo.
- El phasor espectral **no necesita calibración** → esta rama ya sirve para construir
  el cluster de referencia de un polímero.

### FLIM — calibrado con la IRF (actualizado 2026-10-06)

- El `.sdt` tiene **dos detectores** (dos módulos TCSPC sobre el mismo campo): 3,1 M y
  63,3 M fotones. La primera versión de `io_crudo` leía el más débil; ahora
  `phasores_desde_sdt` elige por defecto el de más fotones (`listar_detectores_sdt` los
  enumera).
- Phasor crudo (detector 1): g = 0,016, s = 0,668 → **fuera del semicírculo universal**,
  por el desfase y la de-modulación de la IRF.
- **Calibración con IRF como SPCImage** (`calibrar_irf=True`, IRF sintética del flanco de
  subida): g = 0,470, s = 0,476 → **τφ = 2,69 ns, τm = 2,95 ns**. En el detector 0 da
  τφ = 2,88 ns, τm = 3,34 ns. τφ < τm → emisión no monoexponencial. Los dos detectores
  probablemente miden bandas de emisión distintas (pregunta 13 de `PREGUNTAS_DATOS.md`).
- El método anterior («pico → bin 0», `corregir_desfase=True`) daba τφ ≈ 2,3 ns: en
  simulación subestima el lifetime ~15 %, mientras que la IRF del flanco queda en +2 a
  +6 %. Detalle en `docs/CALIBRACION_FLIM.md`.
- Hay muy pocos fotones por píxel (mediana 2, p99 78). El filtrado del phasor de phasorpy
  compacta la nube 3×, pero todavía no está en el pipeline (`docs/AUDITORIA.md`, hallazgo
  11).
- Una IRF medida en la sesión, o un fluoróforo de referencia, elimina la única
  aproximación que queda.

### z-stack — descartado

`phasor_from_signal` (vía `signal_from_czi`) lo rechaza por no tener eje espectral
creciente. `io_crudo` propaga ese `ValueError` con mensaje claro. Es un z-stack de
intensidad, no entra al análisis de phasores.

## Pendientes para el equipo (bloquean la clasificación)

1. **IRF medida por sesión** (reflexión en el cubreobjetos) o imagen de referencia FLIM
   (fluoróforo + lifetime en ns). Sin ninguna de las dos se usa la IRF sintética.
   Además: el filtro de emisión de cada detector.
2. **Set de calibración de los 6 polímeros** (PET, HDPE, PVC, LDPE, PP, PS) envejecidos
   según `docs/DECISION_CALIBRACION.md`, en el mismo formato. Sin esto no hay contra
   qué clasificar; con estos 3 archivos solo se ubica PE en cada plano.
3. Si se quiere **fusión por píxel**: adquirir FLIM y espectral del mismo campo con la
   misma grilla, o dar la transformación de registro.
4. Confirmar código SPI exacto del "PE" (¿HDPE ♴ o LDPE ♶?).
