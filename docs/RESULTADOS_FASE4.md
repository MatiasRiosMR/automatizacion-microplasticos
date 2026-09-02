# RESULTADOS_FASE4.md — plugin de napari

Fecha: 2026-09-02. Probado en el entorno `napari-mp-env` (conda, Python 3.12,
napari 0.9.0, phasorpy 0.12, PyQt6).

## Por qué un entorno aparte

El código del proyecto depende de **`phasorpy >= 0.12`** (decisión de la Fase 0:
`phasorpy.lifetime`, `phasorpy.component`). `phasorpy >= 0.5` exige **Python ≥ 3.11** y
`>= 0.12` exige **≥ 3.12**. Los entornos existentes con napari (`napari-flim`,
`mnp-phasor-env`) están en Python 3.10 o traen phasorpy 0.4, incompatibles. Por eso la
Fase 4 vive en un entorno dedicado:

```bash
conda create -n napari-mp-env python=3.12
conda activate napari-mp-env
pip install -e ".[dev,napari]"
```

El paquete base (Fases 1–3) sigue corriendo en cualquier entorno con `phasorpy >= 0.12`;
solo la Fase 4 necesita además napari + Qt.

## Qué se agregó

`src/napari_mp_classifier/napari_integracion/`:

| Archivo | Contenido |
|---|---|
| `napari.yaml` | Manifiesto npe2: dos widgets (`Clasificar microplásticos`, `Diagrama de phasores`). |
| `_widget.py` | `WidgetClasificador(Container)` — combos para las capas de intensidad y phasor, ruta de calibración, estrategia/confianza/método. Al pulsar "Clasificar" corre `pipeline.analizar_muestra` y agrega la capa `Labels` **`clasificación MP`** coloreada por polímero predicho (`DirectLabelColormap`), con `layer.features` = tabla de ROIs y `layer.metadata["resultado"]` = `ResultadoMuestra`. |
| `_phasor_plot.py` | `PhasorPlotWidget(QWidget)` — diagrama de phasores (matplotlib) con los clusters de referencia (centroide + elipse 2σ) y un punto por ROI. **Back-projection bidireccional**: click en un punto → `selected_label` de la capa; cambiar la ROI seleccionada en el visor → resalta su punto. Selector de plano (FLIM / espectral) para la fusión 4D. |

Registro en `pyproject.toml`: `[project.entry-points."napari.manifest"]` +
`pytest-qt` en el extra `dev`.

## Verificación

`tests/test_napari_integracion.py` — **6 tests**, corren headless
(`QT_QPA_PLATFORM=offscreen`), se saltan enteros si napari no está instalado:

- manifiesto npe2 válido (2 widgets);
- `WidgetClasificador` se construye contra un visor real (`make_napari_viewer`);
- **flujo completo**: cargar capas → clasificar → aparece la capa `clasificación MP` con
  `features` y `metadata["resultado"]`, y el texto de estado reporta las ROIs;
- error de calibración faltante se muestra en el widget sin agregar capa;
- **back-projection en las dos direcciones** (visor→plot y plot→visor);
- `colores_por_label` devuelve RGBA por ROI.

`ejemplos/demo_fase4.py` arma un visor con una muestra sintética, corre el widget y el
phasor plot, y (con display real) guarda capturas. En headless imprime el resumen y
verifica la back-projection.

Total del proyecto: **84 tests en verde** en `napari-mp-env` (78 base + 6 napari).

## Limitaciones

- Las **capturas de pantalla del lienzo de napari** necesitan un contexto OpenGL real;
  en headless (sin display / `offscreen`) se omiten. La lógica del plugin queda cubierta
  por los tests; para ver la GUI hay que abrir napari con un display:
  `conda run -n napari-mp-env python ejemplos/demo_fase4.py`.
- Compatibilidad de colores de `Labels` escrita para napari 0.5–0.9 (`DirectLabelColormap`)
  con fallback a la API vieja (`layer.color`).
- Convivencia con `napari-phasors` en el mismo visor: prevista (plugin independiente), no
  probada todavía (no está instalado en `napari-mp-env`).
- Entrada por capas / `.npz`; la lectura de `.sdt`/`.czi` crudos entra por `io_crudo.py`
  cuando haya datos reales.
