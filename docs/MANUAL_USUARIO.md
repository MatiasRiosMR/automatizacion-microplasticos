# MANUAL_USUARIO.md

> Borrador. Se completa en la Fase 5 con el pipeline end-to-end y la integración napari.

## Instalación

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # desarrollo y tests
# pip install -e ".[napari]"     # además, integración napari (Fase 4)
```

Requiere Python 3.11+.

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

## Uso como CLI

```bash
napari-mp-classifier --version
napari-mp-classifier classify muestra.tif --calibracion calibracion.csv --salida resultados/
```

(El comando `classify` se implementa en la Fase 3.)

## ¿Cuándo una partícula queda "no clasificable"?

Cuando su distancia al cluster de polímero más cercano supera el umbral estadístico
`chi2.ppf(confianza, df=n_features)` (con `confianza=0.99` por defecto). Es el mecanismo
para no asignar polímero a materia orgánica fluorescente (muestras ambientales) ni a
autofluorescencia celular (monocitos / neutrófilos). Subir `confianza` → menos rechazos,
más riesgo de falso positivo; bajarla → más rechazos, más riesgo de perder polímero real
o envejecido. `confianza=None` desactiva el rechazo.
