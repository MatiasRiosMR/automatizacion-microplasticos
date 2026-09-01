"""Extracción de features por ROI.

Estado: **Fase 2** (pendiente). Features previstas por ROI:

- Coordenadas de phasor promedio (mediana robusta) FLIM y/o espectral.
- Intensidad total y media de Nile Red.
- Tamaño (área en px y µm²), excentricidad, solidez, relación de aspecto.
- Dispersión intra-ROI en el plano de phasores (indicador de mezcla / borde).

Se apoya en ``skimage.measure.regionprops`` y en ``phasorpy.phasor.phasor_center``.
"""

from __future__ import annotations


def extraer_features(*args, **kwargs):  # pragma: no cover - Fase 2
    raise NotImplementedError("Features por ROI: Fase 2. Ver docs/PIPELINE.md.")
