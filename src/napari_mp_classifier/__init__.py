"""napari-mp-classifier — clasificación automática de microplásticos por phasores.

Módulo de análisis automatizado de microplásticos (MP) recalcitrantes teñidos con
Nile Red, clasificados contra 6 polímeros de referencia mediante diagramas de phasores
de dos modalidades de microscopía: espectral (λ-stack) y FLIM (dominio temporal).

Ver ``docs/`` para el fundamento científico y las decisiones de diseño.
"""

from __future__ import annotations

__version__ = "0.1.0.dev0"

# Códigos SPI de los 6 polímeros de referencia del póster.
POLIMEROS: tuple[str, ...] = ("PET", "HDPE", "PVC", "LDPE", "PP", "PS")

# Etiqueta reservada para partículas que caen fuera de los clusters conocidos.
NO_CLASIFICABLE: str = "no_clasificable"

from .calibracion import Calibracion  # noqa: E402
from .clasificador import ClasificadorPhasor  # noqa: E402

__all__ = [
    "__version__",
    "POLIMEROS",
    "NO_CLASIFICABLE",
    "Calibracion",
    "ClasificadorPhasor",
]
