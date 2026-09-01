"""Configuración de pytest: hace importable el paquete y el módulo de datos sintéticos."""

import sys
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")  # figuras sin ventana en los tests

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "tests"))


@pytest.fixture(autouse=True)
def _cerrar_figuras():
    """Cierra toda figura de Matplotlib al terminar cada test (evita fugas de memoria)."""
    yield
    import matplotlib.pyplot as plt

    plt.close("all")
