# Spectral unmixing de λ-stacks — opciones en el ecosistema napari/Python

Relevamiento a pedido del equipo (2026-09-01). Contexto: en Fiji/ImageJ el *linear
unmixing* disponible (plugin "Spectral Unmixing", estilo ZEN) anda mal —muy sensible al
ruido, se rompe con más de 3-4 componentes solapados y no maneja bien la autofluorescencia
como componente desconocido—.

## TL;DR

**Sí, hay una opción claramente mejor y encima está alineada con este proyecto:
el unmixing basado en phasores de `phasorpy`**, opcionalmente vía la interfaz de
`napari-phasors`. Es *model-free*, píxel a píxel, robusto al ruido, y admite un
"componente desconocido" para la autofluorescencia. Es la misma familia de métodos que
usa el grupo de Malacrida (*Phasor-Based Multi-Harmonic Unmixing*, Vitrani et al. 2022).

## Opción recomendada — `phasorpy.component` (+ `napari-phasors`)

`phasorpy` (ya es dependencia base del proyecto) trae el módulo
[`phasorpy.component`](https://www.phasorpy.org/docs/stable/api/components/) con
todo el spectral unmixing sobre el diagrama de phasores espectral:

| Función | Qué hace | Nº de componentes |
|---|---|---|
| `phasor_component_fraction(real, imag, comp_real, comp_imag)` | Fracción de 2 componentes por píxel (proyección sobre la recta que los une) | exactamente 2 |
| `phasor_component_fit(mean, real, imag, comp_real, comp_imag, **kw)` | Ajuste por mínimos cuadrados multi-armónico; fracción de cada componente por píxel | ≥ 2 (demostrado con 5 + 1 desconocido) |
| `phasor_component_graphical(real, imag, comp_real, comp_imag, radius=...)` | Unmixing gráfico multi-componente (histograma de fracciones a lo largo de segmentos) | ≥ 2 |
| `phasor_component_mvc(real, imag, comp_real, comp_imag)` | Coordenadas de valor medio (baricéntricas para 3) dentro del polígono de componentes | ≥ 3 |
| `phasor_component_concentration(...)` | Concentración absoluta usando una referencia de concentración conocida | 2 |

**Entrada**: las coordenadas de phasor de los *espectros puros* de cada componente
(Nile Red-MP de cada polímero, autofluorescencia celular, materia orgánica...). Se
obtienen midiendo cada especie por separado —lo mismo que ya necesitamos para la
calibración de los 6 polímeros—.

**Salida**: una imagen de *fracción* (0-1) por componente y por píxel. Multiplicada por la
intensidad da la imagen de intensidad de ese componente.

**Por qué es mejor que el linear unmixing de Fiji**:
- *Model-free*: no asume forma del espectro, trabaja sobre la transformada de Fourier
  (armónicos) del λ-stack completo, no sobre bandas discretas.
- Robusto al ruido: el phasor promedia todo el espectro en 2 números (g, s) por armónico.
- Maneja un **componente desconocido** (autofluorescencia) sin medirlo: Vitrani et al.
  (2022) demuestran hasta 5 componentes conocidos + 1 desconocido usando 2 armónicos.
- Píxel a píxel y sin inversión de matriz mal condicionada.
- Es **el mismo espacio** (phasores espectrales) donde ya vamos a clasificar → el unmixing
  y la clasificación comparten representación.

**Interfaz napari**: `napari-phasors` lista *"Component analysis for multi-component
systems"* entre sus features (readthedocs / README). Sirve para hacerlo interactivo sin
escribir GUI. La API programática de `phasorpy.component` la podemos usar directo desde
nuestro módulo.

## Otras opciones evaluadas

| Plugin | Qué hace | Sirve para λ-stacks / este caso |
|---|---|---|
| **napari-PICASSO** (`pip install napari-PICASSO`) | Unmixing *ciego* minimizando la información mutua entre pares sink/source (red neuronal, GPU). `unmixed = sink - Σ αᵢ(sourceᵢ - βᵢ)`. Activamente mantenido (NY Genome Center, napari Plugin Accelerator). | Parcial: pensado para *spillover* entre 2-N canales discretos, no para endmember unmixing de un λ-stack. Útil si el problema es crosstalk NR ↔ un segundo marcador. Necesita GPU. |
| **napari-hsi-analysis** (A. Di Benedetto, v0.3.1, jun-2025) | Data Manager + "Fusion" (combina 2-3 datasets) + UMAP. | Exploración / reducción de dimensionalidad, **no** unmixing cuantitativo. Muy temprano. Nota: su "Fusion" es interesante como referencia para nuestra fusión FLIM+espectral. |
| **napari-musa** (hyperpolimi, v1.0.0, nov-2025) | "Análisis de datasets HSI". Metadatos incompletos en napari-hub; repo no accesible al momento del relevamiento. | A revisar cuando esté el repo. Baja prioridad. |
| **domb-napari** (B. Olifirov) | Toolkit de imagen de fluorescencia (ratiometría, registración...). | No es unmixing espectral. |

## Antecedente metodológico

- Vitrani, Cutrale et al. (2022) *Phasor-Based Multi-Harmonic Unmixing for In-Vivo
  Hyperspectral Imaging*, bioRxiv 2022.03.31.486485 / PubMed 36252561. Requiere solo la
  medición empírica de las especies puras; calcula la fracción de fotones por píxel de
  cada componente; demostrado para 5 componentes + 1 desconocido (autofluorescencia).
  Es la base de `phasorpy.component`.

## Encaje en el proyecto

Se agrega un módulo `src/napari_mp_classifier/desmezcla.py` (Fase 3, opcional para Fase 2
en muestras de fagocitos):

1. Antes de clasificar, en las muestras con autofluorescencia (monocitos / neutrófilos,
   materia orgánica ambiental), correr `phasor_component_fit` con los espectros puros de
   NR-MP + un componente de autofluorescencia.
2. Quedarse con la fracción NR-MP por píxel → máscara/pesos para la segmentación
   (complementa el enmascaramiento de Park et al. 2020) y para el promedio de phasor por
   ROI (features menos contaminados).
3. La clasificación de los 6 polímeros sigue igual, pero sobre señal ya "limpia".

Esto **no reemplaza** la clasificación: unmixing = "cuánto de esto es NR-MP vs fondo";
clasificación = "de qué polímero es ese NR-MP". Son pasos distintos del pipeline.
