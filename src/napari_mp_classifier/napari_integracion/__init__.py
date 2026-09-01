"""Integración con napari (Fase 4).

Plugin propio y liviano —no un fork de ``napari-phasors``— que agrega la capa de
clasificación como ``Labels`` vinculada al phasor plot: click en una partícula ↔ resalta
su punto en el diagrama de phasores y viceversa (back-projection).

Puede convivir con ``napari-phasors`` en el mismo visor. Requiere el extra ``[napari]``
del ``pyproject.toml`` (probablemente sobre Python 3.11/3.12 por Qt).
"""

from __future__ import annotations
