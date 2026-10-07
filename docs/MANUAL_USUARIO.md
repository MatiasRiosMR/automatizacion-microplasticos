# MANUAL_USUARIO.md

Cubre el pipeline end-to-end (librería + CLI + napari), desde archivos crudos `.sdt`/`.czi`
o desde phasores ya calculados, hasta el informe por muestra. Recorrido interactivo: [`../ejemplos/notebook_demo.ipynb`](../ejemplos/notebook_demo.ipynb).

## Instalación

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # desarrollo y tests
```

Requiere Python 3.11+ (por `phasorpy >= 0.12`).

Para el **plugin de napari** (Fase 4) hace falta Python 3.12 + Qt, en un entorno aparte:

```bash
conda create -n napari-mp-env python=3.12 && conda activate napari-mp-env
pip install -e ".[dev,napari]"
napari                          # Plugins → Clasificador de microplásticos por phasores
```

## Leer archivos crudos (`.sdt` FLIM / `.czi` λ-stack)

```python
from napari_mp_classifier.io_crudo import (listar_detectores_sdt, phasores_desde_czi,
                                           phasores_desde_sdt)

listar_detectores_sdt("muestra.sdt")          # fotones por detector (los .sdt de LAMAE traen 2)
flim = phasores_desde_sdt("muestra.sdt", calibrar_irf=True)   # detector con más fotones,
                                                              # calibrado con IRF (SPCImage)
esp = phasores_desde_czi("muestra_lambda.czi")                # el espectral no se calibra
g, s, intensidad = flim                       # flim.fotones = suma por píxel
```

`intensidad` es la **media** por bin/canal; para umbrales en fotones usá `fotones`. FLIM y
espectral vienen de adquisiciones distintas (otra grilla), así que se clasifican por separado
o se fusionan por decisión (`fusion.fusionar_por_decision`). Detalle de la calibración
FLIM: [`CALIBRACION_FLIM.md`](CALIBRACION_FLIM.md).

## Uso como librería — clasificar contra los 6 polímeros

```python
import numpy as np
from napari_mp_classifier import Calibracion, ClasificadorPhasor
from napari_mp_classifier.metricas import evaluar_clasificacion

# 1. Calibración desde un CSV de coordenadas de phasor de polímero conocido
cal = Calibracion.cargar_phasores_csv(
    "datos/calibracion/phasores.csv",
    columnas=["g_flim", "s_flim"],      # o las 4 columnas para fusión FLIM+espectral
    columna_etiqueta="polimero",
)

# 2. Clasificador (QDA + umbral de Hotelling; es la estrategia por defecto)
clf = ClasificadorPhasor(cal, estrategia="centroide", confianza=0.99)
clf.entrenar()

# La calibración se puede guardar completa (centroides, covarianzas, n, metadatos)
cal.guardar_json("calibracion.json")
cal = Calibracion.cargar_json("calibracion.json")

# 3. Predicción sobre partículas nuevas (coordenadas de phasor por ROI)
X = np.array([[0.30, 0.46], [0.65, 0.49], [0.9, 0.9]])
etiquetas, score = clf.predecir_con_score(X)
print(etiquetas)   # p.ej. ['PVC' 'PET' 'no_clasificable']

# 4. Métricas (si hay verdad de terreno)
y_true = np.array(["PVC", "PET", "no_clasificable"])
print(evaluar_clasificacion(y_true, etiquetas).resumen())
```

## Uso como librería — pipeline completo (imagen → partículas clasificadas)

```python
from napari_mp_classifier import Calibracion, analizar_muestra
from napari_mp_classifier.reportes import generar_reporte

# canales: dict con 'intensidad' (2D) + phasores por píxel ('g_flim','s_flim','g_esp','s_esp')
# df_cal: DataFrame de mediciones de calibración (columnas de phasor + 'polimero')
columnas = ["g_flim", "s_flim", "g_esp", "s_esp"]
cal = Calibracion.desde_dataframe(df_cal, columnas=columnas)

resultado = analizar_muestra(
    canales, cal,                # estrategia="centroide" por defecto
    # estrategia="knn", mediciones_calibracion=(X_cal, y_cal),  # knn las necesita
    escala_um_px=0.18,          # opcional, para area_um2
    # mascara_celular=mascara,  # opcional, para muestras de fagocitos (Mo/PMN)
)

print(resultado.conteo_por_polimero())
generar_reporte(resultado, "resultados/", canales=canales)   # CSV + métricas + figuras

from napari_mp_classifier.informe_html import generar_informe_html
generar_informe_html(resultado, canales, "resultados/informe.html", nombre_muestra="M-01")
```

El informe por muestra (HTML interactivo + PDF) se describe en [`INFORMES.md`](INFORMES.md).

## Uso como CLI

```bash
napari-mp-classifier --version
napari-mp-classifier classify muestra.npz \
    --calibracion calibracion.csv --salida resultados/ \
    --confianza 0.99 --escala-um-px 0.18 --nombre M-01
```

- `muestra.npz`: arrays 2D `intensidad` (obligatorio) + `g_flim`/`s_flim`/`g_esp`/`s_esp`
  (al menos un par). La modalidad se deduce de los canales presentes; `--modalidad flim`
  o `--modalidad espectral` la restringe aunque estén los cuatro.
- `--calibracion`: un CSV con una fila por medición (columnas de phasor + `polimero`) o un
  `.json` de `Calibracion.guardar_json` (este último solo con `centroide`/`gmm`).
- `--estrategia`: `centroide` (por defecto), `knn` o `gmm`.
- Escribe en `--salida`: `asignaciones.csv`, `resumen_muestra.md`, métricas, figuras y el
  **informe por muestra** `informe.html` (+ `informe.pdf` si hay Chrome/Chromium).
  `--sin-pdf` omite el PDF y `--sin-informe` omite ambos.

## Uso en napari (Fase 4)

En el entorno `napari-mp-env`:

1. Abrí napari y cargá las imágenes de la muestra como capas: la de **intensidad** de
   Nile Red y las de coordenadas de phasor por píxel (`g_flim`, `s_flim`, `g_esp`,
   `s_esp`; al menos un par).
2. `Plugins → Clasificador de microplásticos por phasores`.
3. En el widget: elegí la capa de intensidad y las de phasor, la ruta del CSV de
   calibración, la estrategia y la confianza. **Clasificar**.
4. Aparece la capa `clasificación MP` (Labels) con las partículas coloreadas por polímero
   predicho; las que caen fuera de los clusters quedan en gris (`no_clasificable`).
5. `Plugins → Diagrama de phasores`: muestra las ROIs sobre los clusters de referencia.
   Click en un punto → selecciona esa partícula en el visor; seleccionar una partícula en
   el visor → resalta su punto (back-projection).

## Separar Nile Red-MP de autofluorescencia (desmezcla)

Para muestras ambientales o de fagocitos, antes de clasificar:

```python
from napari_mp_classifier.desmezcla import fracciones_dos_componentes, enmascarar_por_fraccion

frac_mp = fracciones_dos_componentes(canales["g_esp"], canales["s_esp"],
                                     phasor_mp=(0.33, 0.38),
                                     phasor_autofluorescencia=(0.66, 0.60))
canales_filtrados = dict(canales)
mascara_mp = enmascarar_por_fraccion(frac_mp, umbral=0.4)
canales_filtrados["intensidad"] = canales["intensidad"] * mascara_mp
```

## ¿Cuándo una partícula queda "no clasificable"?

Cuando su distancia de Mahalanobis al cluster asignado supera el umbral de la **región de
predicción de Hotelling** de ese cluster (con `confianza=0.99` por defecto). A diferencia
del χ², este umbral tiene en cuenta que el centroide y la covarianza se estiman con pocas
mediciones: con 10 mediciones por polímero el χ² rechazaba 23 % de polímero real en lugar
del 1 % nominal (`AUDITORIA.md`). Una partícula con phasor inválido (NaN, pocos fotones)
también queda `no_clasificable`. Es el mecanismo
para no asignar polímero a materia orgánica fluorescente (muestras ambientales) ni a
autofluorescencia celular (monocitos / neutrófilos). Subir `confianza` → menos rechazos,
más riesgo de falso positivo; bajarla → más rechazos, más riesgo de perder polímero real
o envejecido. `confianza=None` desactiva el rechazo.

**Elegir `confianza` con datos reales.** El `0.995` que recomendaba `RESULTADOS_FASE5.md`
se midió con el umbral χ², que estaba descalibrado; con Hotelling ya no vale. En el
escenario realista de la auditoría, `0.99` rechaza 1,3 % de polímero pero acepta la mitad
de la autofluorescencia que cae cerca de un polímero. El punto de operación se tiene que
fijar con controles negativos (materia orgánica sin MP) y validación cruzada sobre la
calibración real.

La calibración se hace sobre polímero **envejecido con el estándar** (abrasión + H₂O₂
[+ UV]), no virgen — ver `DECISION_CALIBRACION.md`.
