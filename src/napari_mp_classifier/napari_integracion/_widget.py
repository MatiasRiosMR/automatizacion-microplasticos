"""Widget de napari para clasificar microplásticos sobre una imagen de muestra.

Envuelve :func:`napari_mp_classifier.pipeline.analizar_muestra`: toma las capas de
intensidad y de coordenadas de phasor por píxel, una calibración (CSV de mediciones de
polímero conocido) y agrega al visor una capa ``Labels`` con las partículas coloreadas
por polímero predicho. Las features y el :class:`ResultadoMuestra` quedan en
``layer.features`` y ``layer.metadata`` para el diagrama de phasores
(:mod:`napari_mp_classifier.napari_integracion._phasor_plot`).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from magicgui.widgets import (
    ComboBox,
    Container,
    FloatSlider,
    Label,
    PushButton,
    create_widget,
)

from ..calibracion import Calibracion
from ..pipeline import analizar_muestra
from ..reportes import COLOR_NO_CLASIFICABLE, PALETA_POLIMEROS
from . import NINGUNO

if TYPE_CHECKING:
    import napari

_CANALES_OPCIONALES = ("g_flim", "s_flim", "g_esp", "s_esp")


class WidgetClasificador(Container):
    """Contenedor magicgui con los controles de clasificación.

    Parameters
    ----------
    napari_viewer : napari.Viewer
        El visor en el que se agregará la capa de clasificación.
    """

    def __init__(self, napari_viewer: napari.Viewer) -> None:
        super().__init__()
        self._viewer = napari_viewer

        self._intensidad = create_widget(
            annotation="napari.layers.Image", label="intensidad"
        )
        self._canales = {
            nombre: self._combo_de_imagenes(nombre) for nombre in _CANALES_OPCIONALES
        }
        self._calibracion = create_widget(
            annotation=Path, label="calibración (.csv)", options={"mode": "r", "filter": "*.csv"}
        )
        self._estrategia = ComboBox(
            label="estrategia", choices=["knn", "centroide", "gmm"], value="knn"
        )
        self._confianza = FloatSlider(label="confianza", min=0.5, max=0.999, value=0.99)
        self._metodo_seg = ComboBox(
            label="segmentación", choices=["umbral", "kmeans"], value="umbral"
        )
        self._escala = create_widget(
            value=0.0, annotation=float, label="µm/píxel (0 = sin escala)"
        )
        self._boton = PushButton(label="Clasificar")
        self._estado = Label(value="")

        self._boton.clicked.connect(self._ejecutar)
        self.extend([
            self._intensidad, *self._canales.values(), self._calibracion,
            self._estrategia, self._confianza, self._metodo_seg, self._escala,
            self._boton, self._estado,
        ])
        self._viewer.layers.events.inserted.connect(self._refrescar_combos)
        self._viewer.layers.events.removed.connect(self._refrescar_combos)

    # ------------------------------------------------------------------ helpers
    def _combo_de_imagenes(self, nombre: str) -> ComboBox:
        return ComboBox(label=nombre, choices=self._opciones_imagen, value=NINGUNO)

    def _opciones_imagen(self, _widget=None) -> list[str]:
        import napari

        capas = [c.name for c in self._viewer.layers if isinstance(c, napari.layers.Image)]
        return [NINGUNO, *capas]

    def _refrescar_combos(self, _event=None) -> None:
        for combo in self._canales.values():
            combo.reset_choices()

    def _reunir_canales(self) -> dict[str, np.ndarray]:
        canales = {"intensidad": np.asarray(self._intensidad.value.data, dtype=float)}
        for nombre, combo in self._canales.items():
            if combo.value and combo.value != NINGUNO:
                canales[nombre] = np.asarray(self._viewer.layers[combo.value].data, dtype=float)
        return canales

    def _columnas(self, canales: dict) -> list[str]:
        flim = "g_flim" in canales and "s_flim" in canales
        esp = "g_esp" in canales and "s_esp" in canales
        if flim and esp:
            return ["g_flim", "s_flim", "g_esp", "s_esp"]
        if flim:
            return ["g_flim", "s_flim"]
        if esp:
            return ["g_esp", "s_esp"]
        raise ValueError("Elegí al menos un par de capas de phasor (g y s).")

    # ------------------------------------------------------------------ acción
    def _ejecutar(self) -> None:
        try:
            resultado = self._clasificar()
        except (ValueError, FileNotFoundError, KeyError) as error:
            self._estado.value = f"⚠ {error}"
            return

        self._agregar_capa(resultado)
        n_nc = int((resultado.features["polimero_predicho"] == "no_clasificable").sum())
        self._estado.value = (
            f"{resultado.n_rois} ROIs · {resultado.n_rois - n_nc} clasificadas · "
            f"{n_nc} no clasificables"
        )

    def _clasificar(self):
        if self._intensidad.value is None:
            raise ValueError("Falta la capa de intensidad.")
        canales = self._reunir_canales()
        columnas = self._columnas(canales)

        ruta = Path(self._calibracion.value)
        if not ruta.is_file():
            raise FileNotFoundError(f"No encuentro la calibración: {ruta}")
        df_cal = pd.read_csv(ruta)
        faltan = [c for c in [*columnas, "polimero"] if c not in df_cal.columns]
        if faltan:
            raise ValueError(f"La calibración no tiene las columnas {faltan}.")

        calibracion = Calibracion.desde_dataframe(df_cal, columnas=columnas)
        mediciones = (
            (df_cal[columnas].to_numpy(), df_cal["polimero"].to_numpy())
            if self._estrategia.value == "knn"
            else None
        )
        confianza = self._confianza.value if self._confianza.value < 0.999 else None
        escala = self._escala.value or None

        return analizar_muestra(
            canales, calibracion,
            estrategia=self._estrategia.value,
            confianza=confianza,
            metodo_segmentacion=self._metodo_seg.value,
            escala_um_px=escala,
            mediciones_calibracion=mediciones,
        )

    def _agregar_capa(self, resultado) -> None:
        colores = colores_por_label(resultado.features)
        nombre = "clasificación MP"
        if nombre in self._viewer.layers:
            del self._viewer.layers[nombre]
        capa = self._viewer.add_labels(
            resultado.labels.astype(int), name=nombre,
            metadata={"resultado": resultado},
        )
        capa.features = resultado.features.reset_index()
        aplicar_colores(capa, colores)


def colores_por_label(features: pd.DataFrame) -> dict[int, tuple[float, float, float, float]]:
    """``{label: RGBA}`` según el polímero predicho de cada ROI (:data:`PALETA_POLIMEROS`)."""
    from matplotlib.colors import to_rgba

    colores: dict[int, tuple[float, float, float, float]] = {}
    for label, codigo in zip(features.index, features["polimero_predicho"]):
        hexcol = (
            COLOR_NO_CLASIFICABLE if codigo == "no_clasificable"
            else PALETA_POLIMEROS.get(codigo, "#888888")
        )
        colores[int(label)] = to_rgba(hexcol)
    return colores


def aplicar_colores(capa_labels, colores: dict[int, tuple]) -> None:
    """Pinta una capa ``Labels`` con un color por label (compatible con napari 0.5–0.9)."""
    try:
        from napari.utils.colormaps import DirectLabelColormap

        mapa = {None: (0.0, 0.0, 0.0, 0.0), **colores}
        capa_labels.colormap = DirectLabelColormap(color_dict=mapa)
    except (ImportError, AttributeError, TypeError):  # pragma: no cover - napari < 0.5
        capa_labels.color = colores
        capa_labels.color_mode = "direct"
