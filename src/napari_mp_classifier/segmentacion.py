"""Segmentación: detección de ROIs candidatas a microplástico en imágenes de Nile Red.

Estado: **Fase 2** (pendiente). Enfoque previsto, según antecedentes:

- **K-means** sobre intensidad (± coordenadas de phasor) para separar
  señal / fondo / sombra, siguiendo FIMAP (Ho et al. 2025, IoU 87,7%).
- ``watershed`` + ``threshold`` de ``scikit-image`` para separar partículas en contacto.
- **Enmascaramiento previo** para muestras de fagocitos (monocitos/neutrófilos): aislar
  la señal de NR-MP de la autofluorescencia celular antes de extraer phasores
  (Park et al. 2020).

La salida es una imagen de ``labels`` (enteros, 0 = fondo) compatible con napari.
"""

from __future__ import annotations


def segmentar_kmeans(*args, **kwargs):  # pragma: no cover - Fase 2
    raise NotImplementedError("Segmentación: Fase 2. Ver docs/PIPELINE.md.")
