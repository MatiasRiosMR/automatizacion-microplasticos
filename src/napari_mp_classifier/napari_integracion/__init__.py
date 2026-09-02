"""Integración con napari (Fase 4).

Plugin propio y liviano —no un fork de ``napari-phasors``— con dos widgets:

- :class:`._widget.WidgetClasificador`: corre el pipeline
  (:func:`napari_mp_classifier.pipeline.analizar_muestra`) sobre las capas de la muestra y
  agrega una capa ``Labels`` con las partículas coloreadas por polímero predicho.
- :class:`._phasor_plot.PhasorPlotWidget`: diagrama de phasores con los clusters de
  referencia y las ROIs; **back-projection** bidireccional — seleccionar una partícula en
  el visor resalta su punto en el phasor plot y viceversa.

Puede convivir con ``napari-phasors`` en el mismo visor. Requiere el extra ``[napari]``
del ``pyproject.toml`` (Python 3.11/3.12 por Qt y por ``phasorpy >= 0.12``).
"""

from __future__ import annotations

#: Opción "sin capa" en los combos de selección de canal.
NINGUNO: str = "<ninguno>"

__all__ = ["NINGUNO"]
