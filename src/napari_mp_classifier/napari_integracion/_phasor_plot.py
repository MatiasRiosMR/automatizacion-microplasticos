"""Diagrama de phasores interactivo con back-projection hacia la capa de clasificación.

- Dibuja los clusters de referencia de la calibración (centroide + elipse de covarianza)
  y un punto por ROI, coloreado por polímero predicho.
- **Click en un punto** → selecciona esa ROI en la capa ``Labels`` del visor.
- **Cambiar la ROI seleccionada** en el visor → resalta su punto en el diagrama.

Toma los datos de la capa ``Labels`` que produce
:class:`napari_mp_classifier.napari_integracion._widget.WidgetClasificador`
(el :class:`ResultadoMuestra` guardado en ``layer.metadata["resultado"]``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from qtpy.QtWidgets import QComboBox, QLabel, QVBoxLayout, QWidget

from ..features import matriz_features
from ..reportes import (
    COLOR_NO_CLASIFICABLE,
    ESTILO_PUBLICACION,
    MARCADORES_POLIMEROS,
    PALETA_POLIMEROS,
    _elipse_covarianza,
    _planos_de_columnas,
)

if TYPE_CHECKING:
    import napari

_NOMBRE_CAPA = "clasificación MP"


class PhasorPlotWidget(QWidget):
    """Widget Qt con el diagrama de phasores y la sincronización de selección."""

    def __init__(self, napari_viewer: napari.Viewer) -> None:
        super().__init__()
        self._viewer = napari_viewer
        self._resultado = None
        self._capa = None
        self._puntos_scatter = None
        self._resaltado = None
        self._plano = (0, 1)  # índices de columnas (g, s) del plano dibujado

        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        self._fig = Figure(figsize=(5, 5))
        self._fig.subplots_adjust(left=0.15, right=0.97, top=0.95, bottom=0.12)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._ax = self._fig.add_subplot(111)

        self._selector_plano = QComboBox()
        self._selector_plano.currentIndexChanged.connect(self._redibujar)
        self._info = QLabel("Corré 'Clasificar microplásticos' primero.")

        layout = QVBoxLayout(self)
        layout.addWidget(self._selector_plano)
        layout.addWidget(self._canvas)
        layout.addWidget(self._info)

        self._canvas.mpl_connect("pick_event", self._on_pick)
        self._viewer.layers.events.inserted.connect(self._quiza_enganchar)
        self._viewer.layers.events.removed.connect(self._quiza_enganchar)
        self._quiza_enganchar()

    # ------------------------------------------------------------------ enganche
    def _quiza_enganchar(self, _event=None) -> None:
        if _NOMBRE_CAPA not in self._viewer.layers:
            return
        capa = self._viewer.layers[_NOMBRE_CAPA]
        resultado = capa.metadata.get("resultado")
        if resultado is None:
            return
        self._capa = capa
        self._resultado = resultado
        capa.events.selected_label.connect(self._on_seleccion_capa)

        self._selector_plano.blockSignals(True)
        self._selector_plano.clear()
        self._planos = _planos_de_columnas(resultado.columnas_phasor)
        self._selector_plano.addItems([p[0] for p in self._planos])
        self._selector_plano.blockSignals(False)
        self._redibujar()

    # ------------------------------------------------------------------ dibujo
    def _redibujar(self, *_args) -> None:
        if self._resultado is None or not len(self._resultado.features):
            return
        idx = max(self._selector_plano.currentIndex(), 0)
        _, ig, is_, _es_flim = self._planos[idx]
        self._plano = (ig, is_)

        with __import__("matplotlib").rc_context(ESTILO_PUBLICACION):
            self._ax.clear()
            cal = self._resultado.calibracion
            for polimero in cal.etiquetas:
                centro = np.asarray(cal.centroides[polimero], dtype=float)[[ig, is_]]
                cov = np.asarray(cal.covarianzas[polimero], dtype=float)[np.ix_([ig, is_], [ig, is_])]
                color = PALETA_POLIMEROS.get(polimero, "#888888")
                self._ax.add_patch(_elipse_covarianza(
                    centro, cov, 2.0, facecolor=color, alpha=0.12, edgecolor=color, lw=1.2
                ))
                self._ax.scatter(*centro, s=70, c=color, edgecolors="black", zorder=5,
                                 marker=MARCADORES_POLIMEROS.get(polimero, "o"))
                self._ax.annotate(polimero, centro, textcoords="offset points",
                                  xytext=(6, 5), fontsize=8, fontweight="bold")

            X, _ = matriz_features(self._resultado.features, self._resultado.parametros["modalidad"])
            pred = self._resultado.features["polimero_predicho"].to_numpy()
            colores = [
                COLOR_NO_CLASIFICABLE if p == "no_clasificable"
                else PALETA_POLIMEROS.get(p, "#888888")
                for p in pred
            ]
            self._puntos_scatter = self._ax.scatter(
                X[:, ig], X[:, is_], s=36, c=colores, edgecolors="white",
                linewidths=0.5, picker=5, zorder=6,
            )
            self._resaltado = self._ax.scatter([], [], s=180, facecolors="none",
                                               edgecolors="#111", linewidths=2.0, zorder=7)
            self._ax.set_xlabel(f"g ({self._planos[idx][0]})")
            self._ax.set_ylabel(f"s ({self._planos[idx][0]})")

            centros = cal.matriz_centroides()[:, [ig, is_]]
            todo = np.vstack([centros, X[:, [ig, is_]]])
            lo, hi = np.nanmin(todo, axis=0), np.nanmax(todo, axis=0)
            margen = 0.08 + 0.15 * (hi - lo)
            self._ax.set_xlim(lo[0] - margen[0], hi[0] + margen[0])
            self._ax.set_ylim(lo[1] - margen[1], hi[1] + margen[1])
            self._ax.set_aspect("equal", adjustable="box")
        self._canvas.draw_idle()
        self._info.setText(f"{len(X)} ROIs · click en un punto para seleccionarlo en el visor")

    # ------------------------------------------------------------------ eventos
    def _on_pick(self, event) -> None:
        if event.artist is not self._puntos_scatter or self._capa is None:
            return
        fila = int(event.ind[0])
        label = int(self._resultado.features.index[fila])
        self._capa.selected_label = label
        self._capa.show_selected_label = True
        self._marcar(fila)

    def _on_seleccion_capa(self, _event=None) -> None:
        if self._resultado is None:
            return
        label = int(self._capa.selected_label)
        indice = self._resultado.features.index
        if label in indice:
            self._marcar(int(np.flatnonzero(indice == label)[0]))

    def _marcar(self, fila: int) -> None:
        if self._puntos_scatter is None:
            return
        punto = self._puntos_scatter.get_offsets()[fila]
        self._resaltado.set_offsets([punto])
        self._canvas.draw_idle()
        f = self._resultado.features.iloc[fila]
        self._info.setText(
            f"ROI {self._resultado.features.index[fila]} · {f['polimero_predicho']} · "
            f"score {f['score_rechazo']:.2f}"
        )
