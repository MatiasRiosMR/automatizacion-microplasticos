"""Lectura de imágenes crudas ``.sdt`` / ``.czi`` → coordenadas de phasor por píxel.

Wrapper fino sobre :mod:`phasorpy.io` y :mod:`phasorpy.phasor`. Centraliza el manejo de
ejes y la calibración FLIM para no repetir esa lógica en ``calibracion``, ``segmentacion``
y ``cli``.

Estado: **pendiente hasta que el equipo entregue datos reales** (ver
``docs/PREGUNTAS_DATOS.md``). El diseño previsto:

- ``phasores_desde_sdt(ruta, frecuencia_mhz, referencia_lifetime_ns, ruta_referencia)``
  → ``(g, s, intensidad)`` arrays 2D, calibrados con
  :func:`phasorpy.lifetime.phasor_calibrate`.
- ``phasores_desde_czi(ruta, eje_espectral="C")`` → ``(g, s, intensidad)`` 2D, sin
  calibración (la longitud de onda es absoluta en datos hiperespectrales).
- ``phasores_desde_sdt`` y ``phasores_desde_czi`` comparten la firma de salida para que
  ``fusion`` pueda combinarlos.
"""

from __future__ import annotations

_MENSAJE = (
    "io_crudo se implementa en cuanto haya .sdt/.czi de ejemplo y estén respondidas "
    "las preguntas de docs/PREGUNTAS_DATOS.md (frecuencia FLIM, referencia de "
    "calibración, rango espectral, registro entre modalidades)."
)


def phasores_desde_sdt(*args, **kwargs):  # pragma: no cover - pendiente Fase 0+
    raise NotImplementedError(_MENSAJE)


def phasores_desde_czi(*args, **kwargs):  # pragma: no cover - pendiente Fase 0+
    raise NotImplementedError(_MENSAJE)
