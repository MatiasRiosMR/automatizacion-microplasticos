# Antecedentes relevados

Revisión de antecedentes previa a la escritura de código. Cada referencia se lista con la
decisión de diseño de `napari_mp_classifier` que influyó.

## 1. Sancataldo, Avellone y Vetri (2020) — *Nile Red lifetime reveals microplastic identity*

Environ. Sci.: Processes Impacts, 22(11):2266-2275. https://doi.org/10.1039/D0EM00348D

FLIM + phasores con Nile Red para identificar microplásticos, **solo en dominio temporal**
(sin λ-stack). Antecedente metodológico más directo de la Fig. 2 del póster.

**Influencia:** valida el uso de phasores de FLIM como espacio de clasificación; fija la
modalidad temporal como una de las dos entradas. Su limitación (una sola modalidad) es
justamente lo que este proyecto extiende al sumar la modalidad espectral.

## 2. Meyers et al. (2022)

Sci. Total Environ. 823:153441. https://doi.org/10.1016/j.scitotenv.2022.153441

Modelos de ML sobre datos RGB de Nile Red: 95,8% de exactitud (plástico / no-plástico) y
88,1% (identificación de polímero).

**Influencia:** referencia de métricas a igualar o superar. `metricas.py` reporta
exactitud, precisión, recall y F1 en el mismo formato para permitir comparación directa.

## 3. Ho et al. (2025) — FIMAP

J. Environ. Chem. Eng. 13(5):117944. https://doi.org/10.1016/j.jece.2025.117944

Segmentación con **K-means** (IoU = 87,7%) + clasificación por **vecino más cercano
multivariado** (90% precisión, 90% exactitud, 100% recall, F1 = 94,7%), excluyendo
eficazmente materia orgánica.

**Influencia:** define la estrategia de `segmentacion.py` (K-means para separar
señal / fondo / sombra) y respalda el clasificador de centroide/vecino más cercano como
línea base de `clasificador.py`. Su exclusión de materia orgánica es el objetivo de la
categoría "no clasificable".

## 4. Rermborirak et al. (2025)

J. Hazard. Mater. Adv. 19:100787. https://doi.org/10.1016/j.hazadv.2025.100787

Detección con **YOLOv8** sobre Nile Red, 94,8% mAP@50 para 6 polímeros, en dispositivo
portátil de bajo costo.

**Influencia:** referencia de detección (no de clasificación espectral). Se documenta como
alternativa de segmentación por deep learning si K-means/watershed resultan insuficientes
en muestras ambientales; no se adopta en la línea base por costo de datos etiquetados.

## 5. Meyers et al. (2024)

Environ. Sci. Pollut. Res. https://doi.org/10.1007/s11356-024-35289-0

El **envejecimiento ambiental** de los MP degrada la fiabilidad de clasificadores
calibrados solo con polímero virgen.

**Influencia:** obliga a declarar explícitamente en el diseño (Fase 5) si la calibración
usa solo polímero virgen —limitación declarada— o suma un set de muestras degradadas
artificialmente. El clasificador expone un parámetro de tolerancia de cluster
(`confianza`, nivel de confianza del umbral chi²) pensado para absorber la deriva por
envejecimiento.

## 6. Park et al. (2020)

Front. Immunol. 11:203. https://doi.org/10.3389/fimmu.2020.00203

Protocolos de citometría de imagen en flujo para fagocitosis de MP por células inmunes,
con enmascaramiento y selección de features para aislar la señal de MP fagocitado de la
fluorescencia débil / de fondo.

**Influencia:** define el enmascaramiento de `segmentacion.py` para las muestras de
monocitos (Mo) y neutrófilos (PMN): separar señal de NR-MP de la autofluorescencia
celular antes de extraer phasores por ROI.

## Infraestructura open-source que NO se reimplementa

- **`napari-flim-phasor-plotter`** (Zoccoler & Wetzker) — lee FLIM crudo (.ptu, .sdt,
  .tif, .zarr) y genera phasor plots interactivos en napari.
  https://github.com/zoccoler/napari-flim-phasor-plotter
- **`napari-phasors`** (Pannunzio, Zoccoler, Schuty, Malacrida), basado en `phasorpy` —
  calcula phasores de datos FLIM e hiperespectrales, calibra contra fluoróforos de
  referencia, exporta coordenadas. https://www.napari-hub.org/plugins/napari-phasors
- **`phasorpy`** — librería base de cálculo de phasores FLIM/hiperespectral y lectura de
  formatos crudos. https://www.phasorpy.org

## Nota de posicionamiento

**Ninguno de los antecedentes 1-4 combina FLIM + espectral simultáneamente** (usan una
modalidad o la otra). Esa combinación es la diferenciación real de este proyecto frente al
estado del arte. En la documentación se afirma exactamente eso, no un genérico "no existe
nada parecido".
