"""Fusión de las features de phasor FLIM + espectral en un vector por ROI.

Núcleo diferenciador del proyecto: ningún antecedente (Sancataldo 2020, Meyers 2022,
FIMAP 2025, Rermborirak 2025) combina ambas modalidades. Ver ``docs/ANTECEDENTES.md``.

Estado: **Fase 3** (pendiente). Dos modos, según la respuesta a la pregunta 9 de
``docs/PREGUNTAS_DATOS.md``:

- **Fusión por ROI** (si ``.sdt`` y ``.czi`` están registrados espacialmente):
  vector de 4 features ``[g_flim, s_flim, g_esp, s_esp]`` por partícula. Un único
  clasificador en el espacio 4D.
- **Fusión por decisión** (si las adquisiciones son independientes): se clasifica cada
  modalidad por separado y se combinan las probabilidades/posteriores por polímero
  (producto de expertos), marcando ``"no_clasificable"`` si las modalidades disienten.

La calibración 4D ya está soportada hoy: ``Calibracion.desde_dataframe`` y
``ClasificadorPhasor`` funcionan con ``columnas=["g_flim","s_flim","g_esp","s_esp"]``.
"""

from __future__ import annotations


def fusionar_por_roi(*args, **kwargs):  # pragma: no cover - Fase 3
    raise NotImplementedError("Fusión FLIM+espectral: Fase 3. Ver docs/PIPELINE.md.")
