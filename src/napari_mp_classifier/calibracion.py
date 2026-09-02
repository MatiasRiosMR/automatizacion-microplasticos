"""Calibración: firma de referencia de los 6 polímeros en el espacio de phasores.

Una *calibración* es el conjunto de clusters de referencia (uno por polímero) en el
plano de phasores. Cada cluster se resume por su centroide y su matriz de covarianza,
estimados a partir de mediciones de polímero conocido (virgen y/o envejecido).

El espacio de features es genérico: puede ser FLIM (``g_flim``, ``s_flim``), espectral
(``g_esp``, ``s_esp``) o la fusión de ambos (4 dimensiones). Ver ``docs/FASE_0_EVALUACION.md``.

Fuentes de datos admitidas
--------------------------
1. ``DataFrame`` de coordenadas ya calculadas (una fila por medición).
2. CSV de coordenadas de phasor (formato compatible con ``napari-phasors``).
3. Imágenes crudas ``.sdt`` / ``.czi`` (se delega el cálculo en ``phasorpy`` a través de
   :mod:`napari_mp_classifier.io_crudo`; se implementa cuando el equipo entregue datos).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# Nombres de columna esperados por defecto para cada modalidad.
COLUMNAS_FLIM: tuple[str, str] = ("g_flim", "s_flim")
COLUMNAS_ESPECTRAL: tuple[str, str] = ("g_esp", "s_esp")
COLUMNA_ETIQUETA: str = "polimero"


@dataclass
class Calibracion:
    """Firma de referencia de los polímeros en el espacio de phasores.

    Parameters
    ----------
    centroides : dict[str, numpy.ndarray]
        Centroide (vector de dimensión ``n_features``) de cada polímero.
    covarianzas : dict[str, numpy.ndarray]
        Matriz de covarianza ``(n_features, n_features)`` de cada polímero. Se usa para
        la distancia de Mahalanobis y para dibujar las elipses de confianza en el reporte.
    columnas : list[str]
        Nombres de las features, en orden. Define la dimensión del espacio de phasores.
    n_muestras : dict[str, int]
        Cantidad de mediciones que respaldan cada cluster (para trazabilidad).
    metadatos : dict
        Información libre: frecuencia FLIM (MHz), referencia de calibración, rango
        espectral, si incluye polímero envejecido, etc.

    Notes
    -----
    Los 6 polímeros de referencia son los del póster: PET, HDPE, PVC, LDPE, PP, PS
    (códigos SPI ♳–♸). El fundamento es la respuesta diferencial del colorante Nile Red
    según la polaridad/rigidez de cada matriz polimérica, que se traduce en posiciones
    separables en el plano de phasores (Sancataldo et al. 2020 para FLIM).
    """

    centroides: dict[str, np.ndarray]
    covarianzas: dict[str, np.ndarray]
    columnas: list[str]
    n_muestras: dict[str, int] = field(default_factory=dict)
    metadatos: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ propiedades
    @property
    def etiquetas(self) -> list[str]:
        """Lista ordenada de polímeros presentes en la calibración."""
        return list(self.centroides.keys())

    @property
    def n_features(self) -> int:
        """Dimensión del espacio de phasores (2 para una modalidad, 4 para fusión)."""
        return len(self.columnas)

    def matriz_centroides(self) -> np.ndarray:
        """Devuelve los centroides apilados en un array ``(n_polimeros, n_features)``."""
        return np.vstack([self.centroides[e] for e in self.etiquetas])

    # ------------------------------------------------------------------ constructores
    @classmethod
    def desde_dataframe(
        cls,
        df: pd.DataFrame,
        columnas: list[str] | tuple[str, ...] = COLUMNAS_FLIM,
        columna_etiqueta: str = COLUMNA_ETIQUETA,
        metadatos: dict | None = None,
    ) -> Calibracion:
        """Construye una calibración a partir de un ``DataFrame`` de coordenadas.

        Parameters
        ----------
        df : pandas.DataFrame
            Una fila por medición. Debe tener las columnas de ``columnas`` y la de
            ``columna_etiqueta`` con el código del polímero.
        columnas : sequence of str, optional
            Nombres de las columnas de features, en orden. Por defecto FLIM.
        columna_etiqueta : str, optional
            Columna con el polímero de referencia de cada fila.
        metadatos : dict, optional
            Metadatos de adquisición a guardar en la calibración.

        Returns
        -------
        Calibracion

        Raises
        ------
        ValueError
            Si falta alguna columna, o si algún polímero tiene menos de 2 mediciones
            (no se puede estimar covarianza).
        """
        columnas = list(columnas)
        faltantes = [c for c in [*columnas, columna_etiqueta] if c not in df.columns]
        if faltantes:
            raise ValueError(f"El DataFrame no tiene las columnas: {faltantes}")

        centroides: dict[str, np.ndarray] = {}
        covarianzas: dict[str, np.ndarray] = {}
        n_muestras: dict[str, int] = {}

        for etiqueta, grupo in df.groupby(columna_etiqueta, sort=True):
            valores = grupo[columnas].to_numpy(dtype=float)
            valores = valores[~np.isnan(valores).any(axis=1)]
            if len(valores) < 2:
                raise ValueError(
                    f"El polímero '{etiqueta}' tiene {len(valores)} medición(es) válidas; "
                    "se necesitan al menos 2 para estimar la covarianza del cluster."
                )
            centroides[str(etiqueta)] = valores.mean(axis=0)
            covarianzas[str(etiqueta)] = np.cov(valores, rowvar=False)
            n_muestras[str(etiqueta)] = len(valores)

        return cls(
            centroides=centroides,
            covarianzas=covarianzas,
            columnas=columnas,
            n_muestras=n_muestras,
            metadatos=metadatos or {},
        )

    @classmethod
    def cargar_phasores_csv(
        cls,
        ruta: str | Path,
        columnas: list[str] | tuple[str, ...] = COLUMNAS_FLIM,
        columna_etiqueta: str = COLUMNA_ETIQUETA,
        **kwargs_read_csv,
    ) -> Calibracion:
        """Carga una calibración desde un CSV de coordenadas de phasor.

        Compatible con exportaciones de ``napari-phasors`` siempre que se indiquen los
        nombres de columna correctos vía ``columnas`` y ``columna_etiqueta``.

        Parameters
        ----------
        ruta : str or pathlib.Path
            Ruta al archivo CSV.
        columnas, columna_etiqueta : sequence of str, str
            Ver :meth:`desde_dataframe`.
        **kwargs_read_csv
            Se pasan tal cual a :func:`pandas.read_csv`.

        Returns
        -------
        Calibracion
        """
        df = pd.read_csv(ruta, **kwargs_read_csv)
        return cls.desde_dataframe(df, columnas=columnas, columna_etiqueta=columna_etiqueta)

    # ------------------------------------------------------------------ persistencia
    def a_dataframe(self) -> pd.DataFrame:
        """Serializa los centroides a un ``DataFrame`` (una fila por polímero)."""
        filas = []
        for etiqueta in self.etiquetas:
            fila = {COLUMNA_ETIQUETA: etiqueta, "n_muestras": self.n_muestras.get(etiqueta, 0)}
            fila.update(dict(zip(self.columnas, self.centroides[etiqueta])))
            filas.append(fila)
        return pd.DataFrame(filas)

    def guardar_csv(self, ruta: str | Path) -> None:
        """Guarda los centroides de la calibración en un CSV legible."""
        self.a_dataframe().to_csv(ruta, index=False)
