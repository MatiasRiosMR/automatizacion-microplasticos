"""Lectura de imágenes crudas ``.sdt`` / ``.czi`` → coordenadas de phasor por píxel.

Wrapper fino sobre :mod:`phasorpy.io` y :mod:`phasorpy.phasor`. Centraliza el manejo de
ejes y la calibración FLIM para no repetir esa lógica en ``calibracion``, ``segmentacion``
y ``cli``.

- :func:`phasores_desde_sdt` → phasor FLIM (dominio temporal). Opciones de calibración:

  * ``calibrar_irf=True`` (recomendado): IRF sintética del flanco de subida, como
    SPCImage, o una IRF medida (``irf=``). Ver :mod:`napari_mp_classifier.calibracion_flim`.
  * ``ruta_referencia`` + ``referencia_lifetime_ns``: fluoróforo de lifetime conocido.
  * ``corregir_desfase=True``: rotación «pico → 0». **No recomendado**: subestima el
    lifetime ~15 % (``docs/CALIBRACION_FLIM.md``).

  Sin ninguna de ellas el phasor queda rotado/escalado por la IRF del equipo: sirve para
  segmentar y comparar píxeles entre sí, no para leer un lifetime absoluto.

  Un ``.sdt`` puede traer **varios detectores** (un bloque de datos por módulo TCSPC).
  :func:`listar_detectores_sdt` los enumera; por defecto se usa el de más fotones.
- :func:`phasores_desde_czi` → phasor espectral (λ-stack). No necesita calibración: la
  longitud de onda es absoluta en datos hiperespectrales.

Ambas devuelven un :class:`PhasoresImagen`, que se puede desempacar como
``g, s, intensidad = phasores_desde_...(...)`` (las tres 2D, misma forma) o usar por
atributo cuando hace falta la metadata (frecuencia, armónico, si está calibrado).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class PhasoresImagen:
    """Coordenadas de phasor de una imagen, con su metadata.

    Attributes
    ----------
    g, s : numpy.ndarray, shape (alto, ancho)
        Parte real e imaginaria del phasor por píxel (``real`` / ``imag`` de
        ``phasorpy``). Píxeles sin señal quedan en ``nan``.
    intensidad : numpy.ndarray, shape (alto, ancho)
        Intensidad **media por canal** (temporal o espectral), tal como la devuelve
        :func:`phasorpy.phasor.phasor_from_signal`. Es el canal de intensidad que consume
        la segmentación. Para fotones totales por píxel usar :attr:`fotones`.
    frecuencia_mhz : float or None
        Frecuencia de modulación/repetición del láser (solo FLIM).
    armonico : int
        Armónico del phasor calculado.
    calibrado : bool
        ``True`` si el phasor FLIM se calibró con la IRF o con una referencia de lifetime
        conocido (``metadatos["calibracion"]`` indica cuál). El phasor espectral se marca
        ``True`` (no requiere calibración).
    modalidad : {"flim", "espectral"}
    metadatos : dict
        Información libre del archivo (ejes originales, rango espectral, ruta, etc.).
    """

    g: np.ndarray
    s: np.ndarray
    intensidad: np.ndarray
    frecuencia_mhz: float | None = None
    armonico: int = 1
    calibrado: bool = False
    modalidad: str = "flim"
    metadatos: dict = field(default_factory=dict)

    def __iter__(self) -> Iterator[np.ndarray]:
        """Permite ``g, s, intensidad = phasores_desde_...(...)``."""
        yield self.g
        yield self.s
        yield self.intensidad

    @property
    def forma(self) -> tuple[int, int]:
        """Forma ``(alto, ancho)`` de la imagen."""
        return self.g.shape

    @property
    def fotones(self) -> np.ndarray:
        """Cuentas totales por píxel (``intensidad`` × número de canales)."""
        n = self.metadatos.get("n_bins") or self.metadatos.get("n_canales") or 1
        return self.intensidad * n


# --------------------------------------------------------------------------- utilidades


def _reducir_a_2d(datos: np.ndarray, dims: tuple[str, ...], eje: str) -> tuple[np.ndarray, int]:
    """Deja el array en 3D ``(alto, ancho, eje)`` sumando los ejes sobrantes.

    Los ``.sdt`` de Becker & Hickl suelen traer ejes de detector/canal (``Q``, ``C``)
    además de ``Y X H``; se integran (suma) porque acá interesa un único mapa de phasor.
    """
    idx_eje = dims.index(eje)
    ejes_espaciales = [i for i, d in enumerate(dims) if d in ("Y", "X")]
    if len(ejes_espaciales) != 2:
        raise ValueError(f"No se encontraron ejes espaciales Y, X en {dims!r}.")
    ejes_sobrantes = tuple(
        i for i in range(datos.ndim) if i != idx_eje and i not in ejes_espaciales
    )
    if ejes_sobrantes:
        datos = datos.sum(axis=ejes_sobrantes)
        # Recalcular índices tras el colapso.
        dims = tuple(d for i, d in enumerate(dims) if i not in ejes_sobrantes)
        idx_eje = dims.index(eje)
    # Ordenar a (Y, X, eje).
    orden = [dims.index("Y"), dims.index("X"), idx_eje]
    return np.transpose(datos, orden), 2


def _phasor_desde_senal(
    senal: np.ndarray, eje: int, armonico: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from phasorpy.phasor import phasor_from_signal

    media, real, imag = phasor_from_signal(senal, axis=eje, harmonic=armonico)
    return np.asarray(media), np.asarray(real), np.asarray(imag)


# ------------------------------------------------------------------------------- FLIM


def _leer_bloques_sdt(ruta: Path) -> list:
    """Lee todos los bloques de datos (detectores) de un ``.sdt``."""
    from phasorpy.io import signal_from_sdt

    bloques = []
    for indice in range(16):
        try:
            bloques.append(signal_from_sdt(str(ruta), index=indice))
        except TypeError:  # lector sin parámetro ``index`` (un solo bloque)
            return [signal_from_sdt(str(ruta))]
        except (IndexError, ValueError, KeyError):
            break
    if not bloques:
        bloques.append(signal_from_sdt(str(ruta)))
    return bloques


def listar_detectores_sdt(ruta: str | Path) -> list[dict]:
    """Enumera los detectores (bloques de datos) de un ``.sdt`` con sus fotones.

    Los ``.sdt`` de Becker & Hickl guardan un bloque por módulo TCSPC; con dos detectores
    simultáneos hay dos imágenes del mismo campo, típicamente en bandas de emisión
    distintas.

    Parameters
    ----------
    ruta : str or pathlib.Path
        Archivo ``.sdt``.

    Returns
    -------
    list of dict
        Una entrada por detector: ``indice``, ``fotones`` (total), ``forma`` y
        ``frecuencia_mhz``.
    """
    return [
        {"indice": i, "fotones": float(np.asarray(b.values).sum()), "forma": tuple(b.shape),
         "frecuencia_mhz": float(b.attrs.get("frequency", 0.0)) or None}
        for i, b in enumerate(_leer_bloques_sdt(Path(ruta)))
    ]


def phasores_desde_sdt(
    ruta: str | Path,
    *,
    frecuencia_mhz: float | None = None,
    armonico: int = 1,
    detector: int | None = None,
    calibrar_irf: bool = False,
    irf: np.ndarray | None = None,
    ruta_referencia: str | Path | None = None,
    referencia_lifetime_ns: float | None = None,
    corregir_desfase: bool = False,
) -> PhasoresImagen:
    """Phasor FLIM de un ``.sdt`` (Becker & Hickl).

    Parameters
    ----------
    ruta : str or pathlib.Path
        Archivo ``.sdt`` de la muestra.
    frecuencia_mhz : float or None, optional
        Frecuencia de repetición del láser. Si es ``None`` se toma la del encabezado.
    armonico : int, optional
        Armónico del phasor. Por defecto ``1``.
    detector : int or None, optional
        Índice del bloque de datos (detector) a usar. ``None`` (por defecto) elige el de
        más fotones. Ver :func:`listar_detectores_sdt`.
    calibrar_irf : bool, optional
        Calibra con la IRF como SPCImage: ``irf`` si se pasa; si no, una IRF sintética del
        flanco de subida del decaimiento de los píxeles más brillantes (10 %).
        **Recomendado** cuando no hay referencia de lifetime. Por defecto ``False``.
    irf : numpy.ndarray, shape (n_bins,), optional
        IRF medida (reflexión, SHG) en la misma grilla temporal. Implica ``calibrar_irf``.
    ruta_referencia : str or pathlib.Path or None, optional
        ``.sdt`` de un fluoróforo de lifetime **conocido y monoexponencial**. Junto con
        ``referencia_lifetime_ns`` calibra con :func:`phasorpy.lifetime.phasor_calibrate`.
        Tiene prioridad sobre la IRF.
    referencia_lifetime_ns : float or None, optional
        Lifetime de la referencia, en ns.
    corregir_desfase : bool, optional
        Rota el eje temporal para llevar el máximo del decaimiento al canal 0. Aproximación
        **no recomendada** (subestima ~15 % el lifetime); se conserva por compatibilidad.
        ``calibrado`` sigue en ``False``.

    Returns
    -------
    PhasoresImagen
        Con ``modalidad="flim"``. ``metadatos`` incluye ``detector``,
        ``fotones_por_detector`` y ``calibracion`` (``"referencia"``, ``"irf_medida"``,
        ``"irf_flanco"``, ``"desfase"`` o ``None``).

    Raises
    ------
    ValueError
        Si el archivo no tiene eje temporal, no se puede determinar la frecuencia o el
        índice de ``detector`` no existe.

    Notes
    -----
    La calibración con IRF corrige fase y modulación instrumentales: el phasor calibrado
    es ``z / z_IRF``. Validado sobre decaimientos simulados con error de 2–6 % en el
    lifetime de fase entre 1 y 4 ns (``docs/CALIBRACION_FLIM.md``).
    """
    ruta = Path(ruta)
    bloques = _leer_bloques_sdt(ruta)
    fotones = [float(np.asarray(b.values).sum()) for b in bloques]
    if detector is None:
        detector = int(np.argmax(fotones))
    if not 0 <= detector < len(bloques):
        raise ValueError(f"{ruta.name}: no existe el detector {detector} (hay {len(bloques)}).")
    senal = bloques[detector]
    dims = tuple(senal.dims)
    if "H" not in dims:
        raise ValueError(f"{ruta.name}: el .sdt no tiene eje temporal 'H' (dims={dims}).")

    frecuencia = frecuencia_mhz or float(senal.attrs.get("frequency", 0.0)) or None
    if frecuencia is None:
        raise ValueError(
            f"{ruta.name}: no se pudo determinar la frecuencia FLIM; pasá frecuencia_mhz."
        )

    cubo, eje = _reducir_a_2d(np.asarray(senal.values), dims, "H")
    hay_referencia = ruta_referencia is not None and referencia_lifetime_ns is not None
    calibracion = None

    if corregir_desfase and not hay_referencia and not (calibrar_irf or irf is not None):
        pico = int(np.argmax(cubo.sum(axis=(0, 1))))
        cubo = np.roll(cubo, -pico, axis=eje)
        calibracion = "desfase"

    intensidad, real, imag = _phasor_desde_senal(cubo, eje, armonico)
    info_irf = None

    if hay_referencia:
        from phasorpy.io import signal_from_sdt
        from phasorpy.lifetime import phasor_calibrate

        ref = signal_from_sdt(str(Path(ruta_referencia)))
        cubo_ref, eje_ref = _reducir_a_2d(np.asarray(ref.values), tuple(ref.dims), "H")
        media_ref, real_ref, imag_ref = _phasor_desde_senal(cubo_ref, eje_ref, armonico)
        real, imag = phasor_calibrate(
            real, imag, media_ref, real_ref, imag_ref,
            frequency=frecuencia, lifetime=referencia_lifetime_ns, harmonic=armonico,
        )
        calibracion = "referencia"
    elif calibrar_irf or irf is not None:
        from .calibracion_flim import calibrar_con_irf, irf_desde_flanco

        if irf is None:
            total = cubo.sum(axis=eje)
            brillo = total >= np.percentile(total, 90)
            irf = irf_desde_flanco(cubo[brillo].sum(axis=0))
            calibracion = "irf_flanco"
        else:
            calibracion = "irf_medida"
        real, imag, info_irf = calibrar_con_irf(real, imag, irf, frecuencia, armonico)

    return PhasoresImagen(
        g=np.asarray(real, dtype=float),
        s=np.asarray(imag, dtype=float),
        intensidad=np.asarray(intensidad, dtype=float),
        frecuencia_mhz=float(frecuencia),
        armonico=armonico,
        calibrado=calibracion in ("referencia", "irf_flanco", "irf_medida"),
        modalidad="flim",
        metadatos={
            "ruta": str(ruta),
            "dims_original": dims,
            "n_bins": int(dict(zip(dims, senal.shape))["H"]),
            "detector": detector,
            "fotones_por_detector": fotones,
            "calibracion": calibracion,
            "irf": info_irf,
            "referencia": str(ruta_referencia) if ruta_referencia else None,
            "desfase_corregido": calibracion == "desfase",
        },
    )


# --------------------------------------------------------------------------- espectral


def phasores_desde_czi(
    ruta: str | Path,
    *,
    eje_espectral: str = "C",
    armonico: int = 1,
) -> PhasoresImagen:
    """Phasor espectral de un ``.czi`` (Zeiss) de λ-stack.

    Parameters
    ----------
    ruta : str or pathlib.Path
        Archivo ``.czi`` con dimensión espectral (canales equiespaciados en nm).
    eje_espectral : str, optional
        Nombre del eje espectral. Por defecto ``"C"``.
    armonico : int, optional
        Armónico del phasor. Por defecto ``1``.

    Returns
    -------
    PhasoresImagen
        Con ``modalidad="espectral"`` y ``calibrado=True`` (la longitud de onda es
        absoluta; no hace falta calibrar).

    Raises
    ------
    ValueError
        Si el ``.czi`` no es un λ-stack (p. ej. un z-stack): ``phasor_from_signal``
        necesita ≥ 3 canales espectrales equiespaciados.
    """
    from phasorpy.io import signal_from_czi

    ruta = Path(ruta)
    senal = signal_from_czi(str(ruta))
    dims = tuple(senal.dims)
    if eje_espectral not in dims:
        raise ValueError(
            f"{ruta.name}: no tiene eje espectral {eje_espectral!r} (dims={dims})."
        )
    n_canales = senal.shape[dims.index(eje_espectral)]
    if n_canales < 3:
        raise ValueError(
            f"{ruta.name}: solo {n_canales} canales espectrales; se necesitan ≥ 3."
        )

    cubo, eje = _reducir_a_2d(np.asarray(senal.values), dims, eje_espectral)
    intensidad, real, imag = _phasor_desde_senal(cubo, eje, armonico)

    longitudes = None
    if eje_espectral in senal.coords:
        longitudes = np.asarray(senal.coords[eje_espectral].values, dtype=float)

    return PhasoresImagen(
        g=np.asarray(real, dtype=float),
        s=np.asarray(imag, dtype=float),
        intensidad=np.asarray(intensidad, dtype=float),
        frecuencia_mhz=None,
        armonico=armonico,
        calibrado=True,
        modalidad="espectral",
        metadatos={
            "ruta": str(ruta),
            "dims_original": dims,
            "n_canales": int(n_canales),
            "rango_nm": (
                (float(longitudes.min()), float(longitudes.max()))
                if longitudes is not None
                else None
            ),
        },
    )
