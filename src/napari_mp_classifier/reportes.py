"""Reportes y visualización de resultados.

Dos familias de salida:

- **Tablas** (:func:`resultados_a_dataframe`, :func:`guardar_reporte_metricas`): CSV de
  asignaciones por partícula y volcado de métricas a disco.
- **Figuras** de calidad publicación/póster (:func:`figura_phasores`,
  :func:`figura_matriz_confusion`, :func:`figura_metricas_por_clase`,
  :func:`figura_comparacion`): diagrama de phasores con los clusters de referencia y las
  partículas clasificadas, matriz de confusión, barras de métricas por polímero y
  comparación entre modalidades/estrategias, y overlay de la imagen de muestra con las
  ROIs segmentadas y etiquetadas (:func:`figura_segmentacion`). Todas comparten paleta,
  tipografía y estilo (:data:`ESTILO_PUBLICACION`) y se guardan en varios formatos con
  :func:`guardar_figura`.

Pendiente (Fase 3): resumen estadístico por muestra, informe HTML/PDF unificado.

Todo reporte de resultados incluye **siempre** las métricas estándar de
:mod:`napari_mp_classifier.metricas`, no solo el CSV de asignaciones
(requisito de documentación del proyecto).

Notes
-----
La paleta categórica de los 6 polímeros (:data:`PALETA_POLIMEROS`) está en orden fijo y
validada para daltonismo con el método de la *skill* ``dataviz`` (bandas de luminosidad,
piso de croma, separación CVD por pares adyacentes). Como en un diagrama de phasores hay
6 clases y el color por sí solo no separa las 6 con seguridad para todos los tipos de
daltonismo, se agrega **codificación secundaria**: un marcador distinto por polímero
(:data:`MARCADORES_POLIMEROS`), etiquetas directas sobre cada cluster y elipses de
covarianza. Posición + forma + etiqueta portan la identidad; el color es refuerzo.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import NO_CLASIFICABLE
from .metricas import ReporteClasificacion

# --------------------------------------------------------------------------- estética

#: Paleta categórica en orden fijo para los 6 polímeros del póster (códigos SPI ♳–♸).
#: Validada para daltonismo (skill ``dataviz``, pares adyacentes). No reordenar.
PALETA_POLIMEROS: dict[str, str] = {
    "PET": "#2a78d6",   # azul
    "HDPE": "#eb6834",  # naranja
    "PVC": "#1baf7a",   # aqua
    "LDPE": "#eda100",  # amarillo
    "PP": "#e87ba4",    # magenta
    "PS": "#008300",    # verde
}

#: Marcador por polímero — codificación secundaria (independiente del color) para que la
#: identidad se lea también en escala de grises / impresión / daltonismo.
MARCADORES_POLIMEROS: dict[str, str] = {
    "PET": "o", "HDPE": "s", "PVC": "^", "LDPE": "D", "PP": "v", "PS": "P",
}

#: Símbolo de reciclaje SPI de cada polímero (para leyendas y ejes).
SIMBOLOS_SPI: dict[str, str] = {
    "PET": "♳", "HDPE": "♴", "PVC": "♵",
    "LDPE": "♶", "PP": "♷", "PS": "♸",
}

#: Color de las partículas rechazadas (``"no_clasificable"``: materia orgánica,
#: autofluorescencia). Gris neutro, sin hue, para que no compita con los polímeros.
COLOR_NO_CLASIFICABLE: str = "#6f6d67"

#: ``rcParams`` de Matplotlib para todas las figuras del módulo. Estilo sobrio de
#: publicación: sin marco superior/derecho, grilla tenue, tipografía sans del sistema.
ESTILO_PUBLICACION: dict = {
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.edgecolor": "#c3c2b7",
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#e1e0d9",
    "grid.linewidth": 0.6,
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
}


def resultados_a_dataframe(
    X: np.ndarray,
    etiquetas: np.ndarray,
    score: np.ndarray,
    columnas: list[str],
    ids: np.ndarray | None = None,
) -> pd.DataFrame:
    """Arma el CSV de asignaciones: una fila por partícula.

    Parameters
    ----------
    X : numpy.ndarray, shape (n, d)
        Coordenadas de phasor de cada partícula.
    etiquetas : array-like of str, shape (n,)
        Polímero asignado o ``"no_clasificable"``.
    score : array-like, shape (n,)
        Score de rechazo (distancia al cluster). Ver
        :meth:`ClasificadorPhasor.predecir_con_score`.
    columnas : list of str
        Nombres de las columnas de ``X``.
    ids : array-like, optional
        Identificador de cada partícula/ROI. Si es ``None`` se numera ``0..n-1``.

    Returns
    -------
    pandas.DataFrame
    """
    n = len(X)
    if ids is None:
        ids = np.arange(n)
    tabla = pd.DataFrame(X, columns=columnas)
    tabla.insert(0, "id", ids)
    tabla["polimero_predicho"] = np.asarray(etiquetas, dtype=str)
    tabla["score_rechazo"] = np.asarray(score, dtype=float)
    return tabla


def guardar_reporte_metricas(reporte: ReporteClasificacion, carpeta: str | Path) -> dict[str, Path]:
    """Escribe el reporte de métricas en disco (texto + CSVs).

    Genera ``metricas_resumen.txt``, ``metricas_por_clase.csv`` y
    ``matriz_confusion.csv`` en ``carpeta``.

    Returns
    -------
    dict[str, pathlib.Path]
        Rutas de los archivos escritos.
    """
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    rutas = {
        "resumen": carpeta / "metricas_resumen.txt",
        "por_clase": carpeta / "metricas_por_clase.csv",
        "matriz_confusion": carpeta / "matriz_confusion.csv",
    }
    rutas["resumen"].write_text(reporte.resumen(), encoding="utf-8")
    reporte.por_clase.to_csv(rutas["por_clase"])
    reporte.matriz_confusion.to_csv(rutas["matriz_confusion"])
    return rutas


# =========================================================================== figuras


def _etiqueta_leyenda(polimero: str) -> str:
    """``"♳ PET"`` — símbolo SPI + código, para leyendas y ticks."""
    simbolo = SIMBOLOS_SPI.get(polimero)
    return f"{simbolo} {polimero}" if simbolo else polimero


def _planos_de_columnas(columnas: list[str]) -> list[tuple[str, int, int, bool]]:
    """Descompone ``columnas`` en planos 2D graficables.

    Cada plano es ``(nombre, idx_g, idx_s, es_flim)``: para una modalidad simple hay un
    plano; para la fusión (4D) hay dos (FLIM y espectral), que se dibujan uno al lado del
    otro. ``es_flim`` controla si se traza el semicírculo universal (FLIM) o el arco de la
    circunferencia unidad (espectral) como referencia.
    """
    nombres = [c.lower() for c in columnas]
    if len(columnas) == 2:
        es_flim = any("flim" in n for n in nombres) or not any("esp" in n for n in nombres)
        etiqueta = "FLIM" if es_flim else "espectral"
        return [(etiqueta, 0, 1, es_flim)]
    if len(columnas) == 4:
        return [("FLIM", 0, 1, True), ("espectral", 2, 3, False)]
    raise ValueError(
        f"Solo se saben graficar 2 o 4 columnas (modalidad simple o fusión); recibí {columnas}."
    )


def _dibujar_referencia_phasor(ax, es_flim: bool) -> None:
    """Traza el semicírculo universal (FLIM) o la circunferencia unidad (espectral)."""
    import matplotlib.pyplot as plt  # noqa: F401  (asegura backend cargado)
    from matplotlib.patches import Arc

    if es_flim:
        # Semicírculo universal: monoexponenciales, centro (0.5, 0), radio 0.5.
        ax.add_patch(Arc((0.5, 0.0), 1.0, 1.0, theta1=0.0, theta2=180.0,
                         color="#b8b7b0", lw=1.2, zorder=1))
        ax.plot([0.0, 1.0], [0.0, 0.0], color="#b8b7b0", lw=1.2, zorder=1)
    else:
        # Phasores espectrales: se referencia la circunferencia unidad completa.
        ax.add_patch(Arc((0.0, 0.0), 2.0, 2.0, theta1=0.0, theta2=360.0,
                         color="#b8b7b0", lw=1.2, ls=(0, (4, 3)), zorder=1))


def _limites_plano(
    centros: list[np.ndarray],
    covs: list[np.ndarray],
    puntos: np.ndarray,
    n_sigmas: float,
    margen: float = 0.06,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Límites ``(xlim, ylim)`` que encuadran los clusters de referencia y las partículas.

    Toma el rectángulo que cubre los centroides ± ``n_sigmas`` desvíos y el percentil
    2–98 de las partículas, y le suma un margen relativo. Evita que la circunferencia de
    referencia (unidad, en phasores espectrales) domine la escala y deje los datos
    apretados en una esquina.
    """
    xs, ys = [], []
    for c, cov in zip(centros, covs):
        ext = n_sigmas * np.sqrt(np.maximum(np.diag(np.asarray(cov, dtype=float)), 0.0))
        xs += [c[0] - ext[0], c[0] + ext[0]]
        ys += [c[1] - ext[1], c[1] + ext[1]]
    if len(puntos):
        xs += list(np.percentile(puntos[:, 0], [2, 98]))
        ys += list(np.percentile(puntos[:, 1], [2, 98]))
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    ancho, alto = x1 - x0, y1 - y0
    lado = max(ancho, alto, 1e-3)
    px, py = margen * lado + (lado - ancho) / 2, margen * lado + (lado - alto) / 2
    return (x0 - px, x1 + px), (y0 - py, y1 + py)


def _elipse_covarianza(media: np.ndarray, cov: np.ndarray, n_sigmas: float, **kwargs):
    """:class:`~matplotlib.patches.Ellipse` de confianza de una gaussiana 2D.

    Los semiejes son ``n_sigmas * sqrt(autovalores)`` de ``cov`` y el ángulo lo fija el
    autovector dominante. Con ``n_sigmas=2`` la elipse encierra ~86 % de la masa en 2D.
    """
    from matplotlib.patches import Ellipse

    valores, vectores = np.linalg.eigh(np.asarray(cov, dtype=float))
    orden = valores.argsort()[::-1]
    valores, vectores = valores[orden], vectores[:, orden]
    angulo = np.degrees(np.arctan2(vectores[1, 0], vectores[0, 0]))
    ancho, alto = 2.0 * n_sigmas * np.sqrt(np.maximum(valores, 0.0))
    return Ellipse(xy=tuple(media), width=ancho, height=alto, angle=angulo, **kwargs)


def figura_phasores(
    calibracion,
    X: np.ndarray,
    etiquetas_predichas: np.ndarray,
    columnas: list[str],
    etiquetas_reales: np.ndarray | None = None,
    *,
    n_sigmas: float = 2.0,
    resaltar_errores: bool = False,
    titulo: str | None = None,
    max_puntos: int | None = 4000,
    semilla: int = 0,
):
    """Diagrama de phasores: clusters de referencia + partículas clasificadas.

    Para cada polímero de la calibración se dibuja el centroide y su elipse de covarianza
    (``n_sigmas``); encima se grafican las partículas de ``X`` coloreadas y con el marcador
    de su **polímero predicho**, y las rechazadas (``"no_clasificable"``) en gris. Con una
    modalidad simple (2D) es un panel; con la fusión FLIM+espectral (4D) son dos paneles
    (proyección FLIM y proyección espectral).

    Parameters
    ----------
    calibracion : Calibracion
        Firma de referencia (aporta centroides y covarianzas por polímero).
    X : numpy.ndarray, shape (n, d)
        Coordenadas de phasor de las partículas (``d`` = 2 o 4, coherente con ``columnas``).
    etiquetas_predichas : array-like of str, shape (n,)
        Polímero asignado a cada partícula o ``"no_clasificable"``.
    columnas : list of str
        Nombres de las columnas de ``X`` (define modalidad y planos: ver
        :func:`_planos_de_columnas`).
    etiquetas_reales : array-like of str, optional
        Verdad de terreno. Solo se usa si ``resaltar_errores=True``.
    n_sigmas : float, optional
        Tamaño de las elipses de covarianza de referencia. Por defecto ``2.0``.
    resaltar_errores : bool, optional
        Si es ``True`` y hay ``etiquetas_reales``, marca con un aro rojo las partículas mal
        clasificadas (predicho ≠ real).
    titulo : str, optional
        Título de la figura (``suptitle``).
    max_puntos : int or None, optional
        Si ``X`` tiene más de ``max_puntos`` partículas se grafica una submuestra aleatoria
        (solo estético; las métricas se calculan aparte). ``None`` grafica todo.
    semilla : int, optional
        Semilla de la submuestra.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt

    _contorno = [pe.withStroke(linewidth=2.4, foreground="white")]

    X = np.asarray(X, dtype=float)
    pred = np.asarray(etiquetas_predichas, dtype=object)
    reales = None if etiquetas_reales is None else np.asarray(etiquetas_reales, dtype=object)

    if max_puntos is not None and len(X) > max_puntos:
        idx = np.random.default_rng(semilla).choice(len(X), size=max_puntos, replace=False)
        X, pred = X[idx], pred[idx]
        reales = None if reales is None else reales[idx]

    planos = _planos_de_columnas(list(columnas))
    polimeros = calibracion.etiquetas

    with plt.rc_context(ESTILO_PUBLICACION):
        fig, axes = plt.subplots(
            1, len(planos), figsize=(6.2 * len(planos), 5.6), squeeze=False
        )
        axes = axes.ravel()

        for ax, (nombre_plano, ig, is_, es_flim) in zip(axes, planos):
            _dibujar_referencia_phasor(ax, es_flim)

            # Partículas rechazadas primero (quedan de fondo).
            m_nc = pred == NO_CLASIFICABLE
            if m_nc.any():
                ax.scatter(
                    X[m_nc, ig], X[m_nc, is_], s=14, c=COLOR_NO_CLASIFICABLE,
                    marker="x", linewidths=0.8, alpha=0.55, zorder=2,
                    label="no clasificable",
                )

            for polimero in polimeros:
                m = pred == polimero
                if m.any():
                    ax.scatter(
                        X[m, ig], X[m, is_], s=20,
                        c=PALETA_POLIMEROS.get(polimero, "#444444"),
                        marker=MARCADORES_POLIMEROS.get(polimero, "o"),
                        edgecolors="white", linewidths=0.4, alpha=0.85, zorder=3,
                        label=_etiqueta_leyenda(polimero),
                    )

            # Referencia: elipse + centroide + etiqueta directa.
            centros2d, covs2d = [], []
            for polimero in polimeros:
                centro = np.asarray(calibracion.centroides[polimero], dtype=float)
                cov = np.asarray(calibracion.covarianzas[polimero], dtype=float)
                c2 = centro[[ig, is_]]
                cov2 = cov[np.ix_([ig, is_], [ig, is_])]
                centros2d.append(c2)
                covs2d.append(cov2)
                color = PALETA_POLIMEROS.get(polimero, "#444444")
                ax.add_patch(_elipse_covarianza(
                    c2, cov2, n_sigmas, facecolor=color, alpha=0.12,
                    edgecolor=color, lw=1.3, zorder=4,
                ))
                ax.scatter(
                    [c2[0]], [c2[1]], s=90, c=color,
                    marker=MARCADORES_POLIMEROS.get(polimero, "o"),
                    edgecolors="black", linewidths=1.1, zorder=6,
                )
                ax.annotate(
                    _etiqueta_leyenda(polimero), c2, textcoords="offset points",
                    xytext=(8, 6), fontsize=9, fontweight="bold", color="#0b0b0b",
                    zorder=7, path_effects=_contorno,
                )

            xlim, ylim = _limites_plano(centros2d, covs2d, X[:, [ig, is_]], n_sigmas)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)

            if resaltar_errores and reales is not None:
                m_err = (pred != reales) & (reales != NO_CLASIFICABLE)
                if m_err.any():
                    ax.scatter(
                        X[m_err, ig], X[m_err, is_], s=70, facecolors="none",
                        edgecolors="#d03b3b", linewidths=1.3, zorder=5,
                        label="mal clasificada",
                    )

            ax.set_xlabel(f"g ({nombre_plano})")
            ax.set_ylabel(f"s ({nombre_plano})")
            ax.set_aspect("equal", adjustable="box")
            if len(planos) > 1:
                ax.set_title(f"Proyección {nombre_plano}", fontsize=11)

        # Leyenda única, fuera de los ejes.
        manijas, rotulos = axes[0].get_legend_handles_labels()
        vistos: dict[str, object] = {}
        for h, r in zip(manijas, rotulos):
            vistos.setdefault(r, h)
        fig.legend(
            vistos.values(), vistos.keys(), loc="center left",
            bbox_to_anchor=(1.0, 0.5), title="polímero predicho", title_fontsize=9,
        )

        if titulo:
            fig.suptitle(titulo, fontsize=13, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 0.88, 1))
    return fig


def figura_matriz_confusion(
    reporte: ReporteClasificacion,
    *,
    normalizar: str | None = "fila",
    titulo: str | None = "Matriz de confusión",
):
    """Mapa de calor de la matriz de confusión (fila = real, columna = predicho).

    Parameters
    ----------
    reporte : ReporteClasificacion
        Salida de :func:`~napari_mp_classifier.metricas.evaluar_clasificacion`.
    normalizar : {"fila", None}, optional
        ``"fila"`` (por defecto) divide cada fila por su soporte → el color muestra el
        *recall* por clase y la anotación es ``porcentaje`` sobre ``conteo``. ``None`` deja
        los conteos crudos.
    titulo : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    mc = reporte.matriz_confusion
    etiquetas = list(mc.index)
    conteos = mc.to_numpy(dtype=float)

    if normalizar == "fila":
        soporte = conteos.sum(axis=1, keepdims=True)
        datos = np.divide(conteos, soporte, out=np.zeros_like(conteos), where=soporte > 0)
        vmax = 1.0
    elif normalizar is None:
        datos = conteos
        vmax = conteos.max() if conteos.size else 1.0
    else:
        raise ValueError(f"normalizar debe ser 'fila' o None, no {normalizar!r}")

    # Rampa secuencial de un solo hue (azul), skill ``dataviz``.
    rampa = LinearSegmentedColormap.from_list(
        "azules_mp", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
    )
    rotulos = [_etiqueta_leyenda(e) if e != NO_CLASIFICABLE else "no clasif." for e in etiquetas]

    with plt.rc_context(ESTILO_PUBLICACION):
        lado = max(4.8, 0.9 * len(etiquetas) + 2.0)
        fig, ax = plt.subplots(figsize=(lado, lado * 0.9))
        ax.grid(False)
        im = ax.imshow(datos, cmap=rampa, vmin=0.0, vmax=vmax, aspect="equal")

        for i in range(len(etiquetas)):
            for j in range(len(etiquetas)):
                if normalizar == "fila":
                    txt = f"{datos[i, j] * 100:.0f}%\n{int(conteos[i, j])}"
                else:
                    txt = f"{int(conteos[i, j])}"
                ax.text(
                    j, i, txt, ha="center", va="center", fontsize=8,
                    color="white" if datos[i, j] > 0.55 * vmax else "#0b0b0b",
                )

        ax.set_xticks(range(len(etiquetas)), rotulos, rotation=45, ha="right")
        ax.set_yticks(range(len(etiquetas)), rotulos)
        ax.set_xlabel("predicho")
        ax.set_ylabel("real")
        ax.set_title(
            f"{titulo}\nexactitud {reporte.exactitud:.3f} · F1 macro {reporte.f1_macro:.3f}"
            if titulo else None
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel("recall por clase" if normalizar == "fila" else "partículas",
                           rotation=270, labelpad=14)
        cbar.outline.set_visible(False)
        fig.tight_layout()
    return fig


def figura_metricas_por_clase(
    reporte: ReporteClasificacion,
    *,
    titulo: str | None = "Métricas por polímero",
):
    """Barras agrupadas de precisión, recall y F1 por clase.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    por_clase = reporte.por_clase
    clases = list(por_clase.index)
    rotulos = [_etiqueta_leyenda(c) if c != NO_CLASIFICABLE else "no clasif." for c in clases]
    metricas = ["precision", "recall", "f1"]
    colores = ["#2a78d6", "#eb6834", "#1baf7a"]

    x = np.arange(len(clases))
    ancho = 0.26

    with plt.rc_context(ESTILO_PUBLICACION):
        fig, ax = plt.subplots(figsize=(max(6.0, 1.1 * len(clases) + 2.0), 4.6))
        ax.grid(True, axis="y")
        ax.grid(False, axis="x")
        for k, (metrica, color) in enumerate(zip(metricas, colores)):
            valores = por_clase[metrica].to_numpy(dtype=float)
            barras = ax.bar(x + (k - 1) * ancho, valores, ancho, label=metrica, color=color)
            ax.bar_label(barras, fmt="%.2f", fontsize=7, padding=2)

        ax.set_xticks(x, rotulos, rotation=30, ha="right")
        ax.set_ylim(0.0, 1.08)
        ax.set_ylabel("valor")
        ax.axhline(1.0, color="#c3c2b7", lw=0.8)
        ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.02))
        if titulo:
            ax.set_title(titulo, pad=28)
        fig.tight_layout()
    return fig


def figura_comparacion(
    tabla_resumen: pd.DataFrame | list[dict],
    *,
    metricas: tuple[str, ...] = ("exactitud", "F1_polimeros"),
    col_modalidad: str = "modalidad",
    col_estrategia: str = "estrategia",
    titulo: str | None = "Comparación de modalidades y estrategias",
):
    """Barras agrupadas comparando modalidades × estrategias para una o más métricas.

    Pensada para la tabla que arma el demo de Fase 1 (una fila por combinación
    modalidad/estrategia). Un subplot por métrica; en cada uno, los grupos son las
    modalidades y las barras las estrategias.

    Parameters
    ----------
    tabla_resumen : pandas.DataFrame or list of dict
        Debe tener las columnas ``col_modalidad``, ``col_estrategia`` y cada nombre de
        ``metricas``.
    metricas : tuple of str, optional
        Columnas a graficar (un panel cada una).
    col_modalidad, col_estrategia : str, optional
        Nombres de las columnas de agrupamiento.
    titulo : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    df = pd.DataFrame(tabla_resumen).copy()
    modalidades = list(dict.fromkeys(df[col_modalidad]))
    estrategias = list(dict.fromkeys(df[col_estrategia]))
    colores = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

    x = np.arange(len(modalidades))
    ancho = 0.8 / max(len(estrategias), 1)

    with plt.rc_context(ESTILO_PUBLICACION):
        fig, axes = plt.subplots(
            1, len(metricas), figsize=(5.2 * len(metricas), 4.6), squeeze=False,
        )
        axes = axes.ravel()
        for ax, metrica in zip(axes, metricas):
            ax.grid(True, axis="y")
            ax.grid(False, axis="x")
            for k, estrategia in enumerate(estrategias):
                sub = df[df[col_estrategia] == estrategia].set_index(col_modalidad)
                valores = [float(sub.loc[m, metrica]) if m in sub.index else np.nan
                           for m in modalidades]
                desplazamiento = (k - (len(estrategias) - 1) / 2) * ancho
                barras = ax.bar(x + desplazamiento, valores, ancho,
                                label=estrategia, color=colores[k % len(colores)])
                ax.bar_label(barras, fmt="%.3f", fontsize=7, padding=2, rotation=90)
            ax.set_xticks(x, [m.capitalize() for m in modalidades])
            ax.set_ylim(0.0, 1.12)
            ax.set_title(metrica.replace("_", " "))
            ax.set_ylabel("valor")

        manijas, rotulos = axes[0].get_legend_handles_labels()
        fig.legend(manijas, rotulos, loc="upper center", ncol=len(estrategias),
                   bbox_to_anchor=(0.5, 1.0), title="estrategia", title_fontsize=9)
        if titulo:
            fig.suptitle(titulo, y=1.08, fontsize=13, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def figura_segmentacion(
    intensidad: np.ndarray,
    labels: np.ndarray,
    *,
    etiquetas_por_label: dict[int, str] | None = None,
    titulo: str | None = "Segmentación de la muestra",
):
    """Imagen de intensidad con las ROIs segmentadas superpuestas y etiquetadas.

    Parameters
    ----------
    intensidad : numpy.ndarray, shape (alto, ancho)
        Imagen de intensidad de Nile Red (se muestra en escala de grises).
    labels : numpy.ndarray of int, shape (alto, ancho)
        Segmentación (``0`` = fondo).
    etiquetas_por_label : dict[int, str], optional
        Polímero (o ``"no_clasificable"``) asignado a cada label. Si se pasa, cada ROI se
        pinta con el color de su polímero (:data:`PALETA_POLIMEROS`) y se rotula; si no,
        se dibuja solo el contorno de cada ROI.
    titulo : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    from skimage.measure import regionprops
    from skimage.segmentation import find_boundaries

    intensidad = np.asarray(intensidad, dtype=float)
    labels = np.asarray(labels, dtype=int)
    _contorno = [pe.withStroke(linewidth=2.2, foreground="black")]

    with plt.rc_context(ESTILO_PUBLICACION):
        fig, ax = plt.subplots(figsize=(6.4, 6.4))
        ax.grid(False)
        ax.imshow(intensidad, cmap="gray", interpolation="nearest")

        if etiquetas_por_label is None:
            bordes = find_boundaries(labels, mode="outer")
            capa = np.zeros((*labels.shape, 4))
            capa[bordes] = to_rgba("#ff2d2d", 0.9)
            ax.imshow(capa, interpolation="nearest")
        else:
            capa = np.zeros((*labels.shape, 4))
            for region in regionprops(labels):
                codigo = etiquetas_por_label.get(region.label, NO_CLASIFICABLE)
                color = (
                    COLOR_NO_CLASIFICABLE if codigo == NO_CLASIFICABLE
                    else PALETA_POLIMEROS.get(codigo, "#444444")
                )
                capa[labels == region.label] = to_rgba(color, 0.45)
                fila, col = region.centroid
                texto = "NC" if codigo == NO_CLASIFICABLE else codigo
                ax.annotate(
                    texto, (col, fila), color="white", fontsize=7, fontweight="bold",
                    ha="center", va="center", path_effects=_contorno,
                )
            ax.imshow(capa, interpolation="nearest")

        ax.set_xticks([])
        ax.set_yticks([])
        if titulo:
            ax.set_title(titulo)
        fig.tight_layout()
    return fig


def guardar_figura(
    fig,
    ruta_sin_extension: str | Path,
    formatos: tuple[str, ...] = ("png", "pdf"),
    *,
    cerrar: bool = True,
) -> list[Path]:
    """Guarda ``fig`` en varios formatos (PNG rasterizado + PDF/SVG vectorial para póster).

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    ruta_sin_extension : str or pathlib.Path
        Ruta base; se le agrega ``.png``, ``.pdf``, etc. Se crean los directorios padre.
    formatos : tuple of str, optional
        Extensiones a escribir. Por defecto ``("png", "pdf")``.
    cerrar : bool, optional
        Cierra la figura tras guardar (libera memoria). Por defecto ``True``.

    Returns
    -------
    list[pathlib.Path]
        Rutas escritas.
    """
    import matplotlib.pyplot as plt

    ruta = Path(ruta_sin_extension)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    escritas = []
    for formato in formatos:
        destino = ruta.with_suffix(f".{formato}")
        fig.savefig(destino, dpi=300, bbox_inches="tight")
        escritas.append(destino)
    if cerrar:
        plt.close(fig)
    return escritas
