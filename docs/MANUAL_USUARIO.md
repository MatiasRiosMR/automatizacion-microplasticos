# MANUAL_USUARIO.md

> Borrador. Se completa en la Fase 5 con el pipeline end-to-end y la integración napari.

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

# 2. Clasificador
clf = ClasificadorPhasor(cal, estrategia="centroide", confianza=0.99)
clf.entrenar()

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
    canales, cal,
    estrategia="knn",
    mediciones_calibracion=(df_cal[columnas].to_numpy(), df_cal["polimero"].to_numpy()),
    escala_um_px=0.18,          # opcional, para area_um2
    # mascara_celular=mascara,  # opcional, para muestras de fagocitos (Mo/PMN)
)

print(resultado.conteo_por_polimero())
generar_reporte(resultado, "resultados/", canales=canales)   # CSV + métricas + figuras
```

## Uso como CLI

```bash
napari-mp-classifier --version
napari-mp-classifier classify muestra.npz \
    --calibracion calibracion.csv --salida resultados/ \
    --estrategia knn --confianza 0.99 --escala-um-px 0.18
```

- `muestra.npz`: arrays 2D `intensidad` (obligatorio) + `g_flim`/`s_flim`/`g_esp`/`s_esp`
  (al menos un par). La modalidad se deduce de los canales presentes.
- `calibracion.csv`: una fila por medición de calibración, con las columnas de phasor y
  una columna `polimero`.
- Escribe en `--salida` el informe unificado: `asignaciones.csv`, `resumen_muestra.md`,
  métricas y figuras.
- La lectura de `.sdt` / `.czi` crudos se habilita cuando esté `io_crudo.py` (datos reales).

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

Cuando su distancia al cluster de polímero más cercano supera el umbral estadístico
`chi2.ppf(confianza, df=n_features)` (con `confianza=0.99` por defecto). Es el mecanismo
para no asignar polímero a materia orgánica fluorescente (muestras ambientales) ni a
autofluorescencia celular (monocitos / neutrófilos). Subir `confianza` → menos rechazos,
más riesgo de falso positivo; bajarla → más rechazos, más riesgo de perder polímero real
o envejecido. `confianza=None` desactiva el rechazo.
