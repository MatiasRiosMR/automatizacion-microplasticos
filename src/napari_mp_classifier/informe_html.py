"""Informe HTML/PDF por muestra, pensado para el investigador.

:func:`generar_informe_html` toma el :class:`~napari_mp_classifier.pipeline.ResultadoMuestra`
de :func:`~napari_mp_classifier.pipeline.analizar_muestra` y escribe un ``.html``
autocontenido con formato de informe de laboratorio (y, si hay Chrome/Chromium instalado,
el ``.pdf`` equivalente con :func:`exportar_a_pdf`).

Contenido
---------
1. Portada: indicadores clave, barra de composición por polímero y estado del análisis
   («Válido», «Válido con observaciones», «No concluyente»), calculado a partir de
   controles automáticos (tamaño de la calibración, fracción no clasificable, partículas a
   revisar, escala espacial).
2. Muestra y condiciones del análisis.
3. Resultados: composición con IC 95 % (Wilson), imagen con partículas numeradas,
   diagramas de phasores FLIM y espectral, tamaño, score de rechazo, partículas con
   observaciones y tabla por partícula.
4. Verificación (solo si hay verdad de terreno): indicadores y matriz de confusión.
5. Observaciones, método, alcance y limitaciones, firmas.
6. Anexos: calibración de referencia con trazabilidad (SHA-256) y glosario.

En pantalla el informe es interactivo: la imagen de partículas, los diagramas de phasores
(Plotly, cargado desde CDN) y la tabla están vinculados —seleccionar una partícula en
cualquiera la marca en los demás—, cada término técnico tiene una ayuda «?» y los datos se
descargan en CSV. Al imprimir se ocultan los controles y Chrome agrega un pie propio
(«Informe N.º · Página X de Y») en lugar del encabezado del navegador.
"""

from __future__ import annotations

import base64
import hashlib
import html
import io
import json
import platform
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import chi2

from . import NO_CLASIFICABLE, POLIMEROS
from .clasificador import umbral_hotelling
from .glosario import CSS_AYUDA, JS_AYUDA, ayuda
from .reportes import (
    ESTILO_PUBLICACION,
    MARCADORES_POLIMEROS,
    PALETA_POLIMEROS,
)

NC = NO_CLASIFICABLE
CLASES = [*POLIMEROS, NC]
NOMBRE = {"PET": "Tereftalato de polietileno", "HDPE": "Polietileno de alta densidad",
          "PVC": "Policloruro de vinilo", "LDPE": "Polietileno de baja densidad",
          "PP": "Polipropileno", "PS": "Poliestireno", NC: "No clasificable"}
COLOR = {**PALETA_POLIMEROS, NC: "#8a8984"}
TINTA, TINTA2 = "#111111", "#444444"

#: Estilo de figura de los informes: sobrio, de publicación (marco completo, marcas hacia
#: adentro, sin títulos en negrita).
ESTILO_INFORME: dict = {
    **ESTILO_PUBLICACION,
    "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"], "font.size": 8.5,
    "axes.titlesize": 9.5, "axes.titleweight": "normal", "axes.labelsize": 9,
    "axes.labelcolor": TINTA, "axes.edgecolor": TINTA, "axes.linewidth": 0.6,
    "axes.spines.top": True, "axes.spines.right": True, "axes.grid": False,
    "grid.color": "#e8e8e8", "grid.linewidth": 0.5, "xtick.color": TINTA, "ytick.color": TINTA,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 7.5, "svg.fonttype": "none",
}
_AZULES = LinearSegmentedColormap.from_list(
    "azules", ["#f6f9fe", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def _a_svg(fig) -> str:
    """Serializa una figura a SVG embebible y la cierra."""
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    svg = buf.getvalue()
    return svg[svg.find("<svg"):]


def _rotulo(c: str) -> str:
    return "No clasif." if c == NC else c


def _fig_confusion(mc: pd.DataFrame):
    """Matriz de confusión normalizada por fila, con conteos."""
    etiquetas = list(mc.index)
    cnt = mc.to_numpy(float)
    sop = cnt.sum(1, keepdims=True)
    dat = np.divide(cnt, sop, out=np.zeros_like(cnt), where=sop > 0)
    n = len(etiquetas)
    with plt.rc_context(ESTILO_INFORME):
        fig, ax = plt.subplots(figsize=(0.62 * n + 2.4, 0.56 * n + 1.4))
        ax.imshow(dat, cmap=_AZULES, vmin=0, vmax=1)
        for i in range(n):
            for j in range(n):
                if cnt[i, j] == 0:
                    continue
                oscuro = dat[i, j] > 0.55
                ax.text(j, i - 0.12, f"{dat[i, j] * 100:.0f}%", ha="center", va="center",
                        fontsize=8, fontweight="bold", color="white" if oscuro else TINTA)
                ax.text(j, i + 0.24, f"{int(cnt[i, j])}", ha="center", va="center",
                        fontsize=6.5, color="#dfe9f7" if oscuro else TINTA2)
        if NC in etiquetas:
            q = etiquetas.index(NC) - 0.5
            ax.axhline(q, color="white", lw=2.5)
            ax.axvline(q, color="white", lw=2.5)
        rot = [_rotulo(e) for e in etiquetas]
        ax.set_xticks(range(n), rot, rotation=40, ha="right")
        ax.set_yticks(range(n), rot)
        ax.tick_params(length=0)
        ax.set_xlabel("Asignación")
        ax.set_ylabel("Composición real")
        fig.tight_layout()
    return fig


def _fig_tamanos(feats, col, etiqueta):
    """Diámetro por polímero (puntos) con la mediana."""
    clases = [c for c in CLASES if (feats.polimero_predicho == c).any()]
    rng = np.random.default_rng(0)
    with plt.rc_context(ESTILO_INFORME):
        fig, ax = plt.subplots(figsize=(5.4, 0.44 * len(clases) + 1.3))
        ax.grid(True, axis="x")
        ax.grid(False, axis="y")
        y = np.arange(len(clases))[::-1]
        for yi, c in zip(y, clases):
            v = feats.loc[feats.polimero_predicho == c, col].to_numpy(float)
            ax.scatter(v, yi + rng.uniform(-0.17, 0.17, len(v)), s=26, color=COLOR[c],
                       marker=MARCADORES_POLIMEROS.get(c, "o"), edgecolors="white", linewidths=0.6,
                       zorder=3)
            med = np.median(v)
            ax.plot([med, med], [yi - 0.3, yi + 0.3], color=TINTA, lw=1.6, zorder=4)
        ax.set_yticks(y, ["No clasif." if c == NC else c for c in clases])
        ax.set_xlabel(etiqueta)
        fig.tight_layout()
    return fig


def _fig_confianza(feats):
    """Histograma apilado del score de rechazo por polímero."""
    s = feats["score_rechazo"].to_numpy(float)
    s = np.clip(np.where(np.isfinite(s), s, 3.0), 0, 3.0)
    with plt.rc_context(ESTILO_INFORME):
        fig, ax = plt.subplots(figsize=(5.4, 3.3))
        ax.grid(True, axis="y")
        ax.grid(False, axis="x")
        bins = np.linspace(0, 3, 31)
        abajo = np.zeros(len(bins) - 1)
        for c in CLASES:
            v = s[feats.polimero_predicho.to_numpy() == c]
            if len(v):
                hst, _ = np.histogram(v, bins)
                ax.bar(bins[:-1], hst, width=0.094, align="edge", bottom=abajo,
                       color=COLOR[c], zorder=2, label="No clasif." if c == NC else c)
                abajo += hst
        ax.axvspan(0.7, 1.0, color="#000", alpha=0.06, lw=0, zorder=1)
        ax.axvline(1.0, color=TINTA, lw=1.1, ls=(0, (3, 2)), zorder=3)
        ax.text(1.03, ax.get_ylim()[1] * 0.92, "umbral de rechazo", fontsize=7.5,
                color=TINTA2, va="top")
        ax.text(0.85, ax.get_ylim()[1] * 0.92, "zona\nlímite", fontsize=7, color="#444",
                ha="center", va="top")
        ax.set_xlabel("Score de rechazo (distancia al cluster / umbral)  ·  >3 agrupado en 3")
        ax.set_ylabel("Partículas")
        ax.legend(ncol=7, loc="upper center", bbox_to_anchor=(0.5, -0.3), fontsize=7.5,
                  handlelength=0.9, columnspacing=0.9)
        fig.tight_layout()
    return fig


def alertas(feats: pd.DataFrame, labels: np.ndarray, tam_min: int = 8) -> tuple[dict, float]:
    """Observaciones automáticas por partícula.

    Parameters
    ----------
    feats : pandas.DataFrame
        Features por ROI con ``polimero_predicho`` y ``score_rechazo``.
    labels : numpy.ndarray of int
        Segmentación de la muestra.
    tam_min : int, optional
        Área mínima de segmentación (px); por debajo de ``3 * tam_min`` se marca «muy pequeña».

    Returns
    -------
    observaciones : dict[int, list[tuple[str, str]]]
        Por ROI, lista de ``(nivel, texto)`` con nivel ``"err"``, ``"warn"`` o ``"info"``.
        ``"err"``/``"warn"`` marcan partículas que requieren revisión manual: phasor
        inválido, score entre 0,7 y 1 (cerca del umbral) o phasor heterogéneo.
    umbral_dispersion : float
        Dispersión intra-ROI a partir de la cual se marca «phasor heterogéneo».
    """
    disp_cols = [c for c in ("dispersion_flim", "dispersion_esp") if c in feats]
    disp = feats[disp_cols].max(axis=1) if disp_cols else pd.Series(0, index=feats.index)
    umbral_disp = max(0.05, float(np.nanpercentile(disp, 90))) if len(disp) else 0.05
    out = {}
    for lab, f in feats.iterrows():
        a = []
        sc = f["score_rechazo"]
        if not np.isfinite(sc):
            a.append(("err", "phasor no válido"))
        elif 0.7 < sc <= 1.0:
            a.append(("warn", "cerca del umbral"))
        m = labels == int(lab)
        if m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any():
            a.append(("info", "toca el borde"))
        if np.isfinite(disp.loc[lab]) and disp.loc[lab] > umbral_disp:
            a.append(("warn", "phasor heterogéneo"))
        if f.get("solidez", 1) < 0.85:
            a.append(("info", "forma irregular / posible contacto"))
        if f["area_px"] < 3 * tam_min:
            a.append(("info", "muy pequeña"))
        if "polimero_real" in f and f["polimero_real"] != f["polimero_predicho"]:
            a.append(("err", "composición real: " + ("No clasificable" if f['polimero_real'] == NC else f['polimero_real'])))
        out[lab] = a
    return out, umbral_disp


def generar_informe_html(
    resultado,
    canales: dict,
    ruta: str | Path,
    *,
    nombre_muestra: str,
    archivo: str | Path | None = None,
    escala_um_px: float | None = None,
    descripcion: str | None = None,
    nota_escala: str | None = None,
    calibracion_flim: str | None = None,
    exportar_pdf: bool = True,
) -> Path:
    """Escribe el informe HTML (y PDF) de una muestra analizada.

    Parameters
    ----------
    resultado : ResultadoMuestra
        Salida de :func:`~napari_mp_classifier.pipeline.analizar_muestra`. Si incluye
        ``polimero_real`` (verdad de terreno) se agrega la sección de verificación.
    canales : dict[str, numpy.ndarray]
        Canales de la muestra; se usa ``"intensidad"`` para la imagen de partículas.
    ruta : str or pathlib.Path
        Archivo ``.html`` de destino. El PDF se escribe junto, con extensión ``.pdf``.
    nombre_muestra : str
        Identificación de la muestra (título del informe).
    archivo : str or pathlib.Path, optional
        Archivo de datos analizado: su nombre y huella SHA-256 quedan en la trazabilidad y
        la huella forma parte del número de informe.
    escala_um_px : float, optional
        Tamaño de píxel en µm. Si es ``None`` se usa el del análisis; sin escala los
        tamaños se informan en píxeles.
    descripcion : str, optional
        Descripción libre de la muestra.
    nota_escala : str, optional
        Aclaración sobre el origen de la escala (p. ej. ``" (valor nominal)"``); genera una
        observación.
    calibracion_flim : str, optional
        Descripción de cómo se calibró el FLIM (aparece en la ficha y en métodos).
    exportar_pdf : bool, optional
        Si es ``True`` (por defecto) intenta generar el PDF con :func:`exportar_a_pdf`.

    Returns
    -------
    pathlib.Path
        Ruta del HTML escrito.
    """
    from . import __version__ as version_paquete
    from .metricas import evaluar_clasificacion

    feats = resultado.features.copy()
    feats.index = feats.index.astype(int)
    p = resultado.parametros
    cal = resultado.calibracion
    cols = resultado.columnas_phasor
    hay_verdad = "polimero_real" in feats
    escala = escala_um_px or p.get("escala_um_px")
    unidad_area = "µm²" if escala else "px²"
    unidad_lin = "µm" if escala else "px"
    feats["area"] = feats["area_um2"] if escala else feats["area_px"]
    feats["diametro_eq"] = 2 * np.sqrt(feats["area"] / np.pi)
    al, _ = alertas(feats, resultado.labels)
    n = len(feats)
    pred = feats.polimero_predicho
    n_nc = int((pred == NC).sum())
    n_mp = n - n_nc
    conteo = pred.value_counts()
    dom = conteo.drop(NC, errors="ignore")
    lideres = list(dom[dom == dom.max()].index) if len(dom) else []
    dominante = lideres[0] if len(lideres) == 1 else ("empate" if lideres else "—")
    a_revisar = [lab for lab, a in al.items() if any(k in ("warn", "err") for k, _ in a)]

    # ---------- figuras
    conf = p.get("confianza") or 0.99
    f_tam = _a_svg(_fig_tamanos(feats, "diametro_eq", f"Diámetro equivalente ({unidad_lin})")) if n else ""
    f_conf = _a_svg(_fig_confianza(feats)) if n else ""




    # ================================================================ formato
    def coma(v, d=1):
        if v is None or (isinstance(v, (float, np.floating)) and not np.isfinite(v)):
            return "—"
        return f"{v:,.{d}f}".replace(",", " ").replace(".", ",")

    def pct(v, d=1):
        return "—" if not np.isfinite(v) else coma(100 * v, d) + " %"

    def mayus(t):
        return t[:1].upper() + t[1:] if t else t

    def codtxt(c):
        return "No clasif." if c == NC else c

    def cod(c):
        return f"<span class='pol'><i style='background:{COLOR.get(c, '#999')}'></i>{codtxt(c)}</span>"

    def wilson(k, n_, z=1.96):
        if n_ == 0:
            return (np.nan, np.nan)
        ph = k / n_
        den = 1 + z ** 2 / n_
        c = (ph + z ** 2 / (2 * n_)) / den
        h = z * np.sqrt(ph * (1 - ph) / n_ + z ** 2 / (4 * n_ ** 2)) / den
        return max(0.0, c - h), min(1.0, c + h)

    def tabla(cabeceras, filas, num=frozenset(), clase="", id_=""):
        th = "".join(f"<th class='{'r' if i in num else ''}'>{c}</th>" for i, c in enumerate(cabeceras))
        return (f"<table class='{clase}' {f'id={id_!r}' if id_ else ''}><thead><tr>{th}</tr></thead>"
                f"<tbody>{''.join(filas)}</tbody></table>")

    def td(v, num=True):
        return f"<td class='{'r' if num else ''}'>{v}</td>"

    # ================================================================ datos derivados
    huella = (hashlib.sha256(Path(archivo).read_bytes()).hexdigest() if archivo and Path(archivo).exists() else "")
    ahora = datetime.now().astimezone()
    id_informe = f"MP-{ahora:%Y%m%d}-{(huella or nombre_muestra.encode().hex())[:6].upper()}"
    fecha = ahora.strftime("%d/%m/%Y")
    hora = ahora.strftime("%H:%M")
    modalidad = {"fusion": "FLIM + espectral (combinadas)", "flim": "FLIM",
                 "espectral": "Espectral (λ-stack)"}[p["modalidad"]]
    estrategia_txt = {"knn": "k vecinos más cercanos (KNN)", "centroide": "centroide más cercano (Mahalanobis)",
                      "gmm": "mezcla de gaussianas (GMM)"}.get(p["estrategia"], p["estrategia"])
    n_cal = cal.n_muestras
    mp = feats[pred != NC]
    d_q = np.percentile(mp.diametro_eq, [25, 50, 75]) if len(mp) else [np.nan] * 3
    orden_cal = sorted(cal.etiquetas, key=lambda x: CLASES.index(x) if x in CLASES else 99)
    presentes = [c for c in CLASES if conteo.get(c, 0)]
    rango_n = ("—" if not n_cal else str(min(n_cal.values())) if min(n_cal.values()) == max(n_cal.values())
               else f"{min(n_cal.values())}–{max(n_cal.values())}")

    # observaciones y conclusión
    obs = []
    n_cal_min = min(n_cal.values()) if n_cal else 0
    if n == 0:
        obs.append("No se detectaron partículas en la imagen; el análisis no es concluyente.")
    if n_cal_min and n_cal_min < 15:
        obs.append(f"La calibración tiene pocas mediciones por polímero (mínimo {n_cal_min}); el criterio de "
                   "rechazo puede resultar excesivamente estricto.")
    if not cal.metadatos:
        obs.append("La calibración no registra metadatos de adquisición (fecha, equipo, protocolo de envejecimiento).")
    if n and n_nc / n > 0.30:
        obs.append(f"El {pct(n_nc / n, 0)} de las partículas resultó no clasificable; se sugiere verificar la "
                   "matriz de la muestra y la vigencia de la calibración.")
    if a_revisar:
        obs.append(f"{len(a_revisar)} partícula(s) presentan observaciones y requieren verificación visual "
                   "(apartado 2.5).")
    if nota_escala:
        obs.append("La escala espacial empleada" + nota_escala.replace("(", "").replace(")", "").replace(
            "valor", "corresponde a un valor") + ".")
    if not escala:
        obs.append("No se informó la escala espacial; los tamaños se expresan en píxeles.")
    conclusion = "No concluyente" if n == 0 else ("Válido con observaciones" if obs else "Válido")
    est_cls = {"Válido": "ok", "No concluyente": "err"}.get(conclusion, "warn")

    # ================================================================ portada: indicadores + barra
    segmentos = "".join(
        f"<span class='seg-c' style='flex:{int(conteo[c])};background:{COLOR[c]}' "
        f"title='{codtxt(c)}: {int(conteo[c])} partículas ({pct(conteo[c] / n)})'>"
        f"{'<b>' + codtxt(c) + '</b>' if conteo[c] / n >= 0.08 else ''}</span>" for c in presentes) if n else ""
    leyenda_comp = "".join(f"<span>{cod(c)} {int(conteo[c])} <em>({pct(conteo[c] / n, 0)})</em></span>"
                           for c in presentes)
    indicadores = f"""
<div class="kpis">
  <div><span class="v">{n}</span><span class="l">partículas detectadas{ayuda('roi')}</span></div>
  <div><span class="v">{n_mp}<small> {pct(n_mp / n, 0) if n else ''}</small></span><span class="l">asignadas a un polímero</span></div>
  <div><span class="v">{n_nc}</span><span class="l">no clasificables{ayuda('no_clasif')}</span></div>
  <div><span class="v">{coma(d_q[1])}<small> {unidad_lin}</small></span><span class="l">diámetro mediano{ayuda('diametro')}</span></div>
  <div><span class="v">{coma(mp.area.sum(), 0)}<small> {unidad_area}</small></span><span class="l">área total de microplástico{ayuda('area_total')}</span></div>
</div>
<div class="comp" role="img" aria-label="Composición de la muestra">{segmentos}</div>
<div class="comp-ley">{leyenda_comp}</div>"""

    # ================================================================ ficha
    ficha = f"""
<dl class="ficha">
  <div><dt>Muestra</dt><dd>{html.escape(nombre_muestra)}</dd></div>
  <div><dt>Archivo</dt><dd>{html.escape(Path(archivo).name) if archivo else '—'}</dd></div>
  <div class="ancho"><dt>Descripción</dt><dd>{html.escape(descripcion or '—')}</dd></div>
  <div><dt>Modalidad{ayuda('fusion' if p['modalidad'] == 'fusion' else ('flim' if p['modalidad'] == 'flim' else 'espectral'))}</dt><dd>{modalidad}</dd></div>
  <div><dt>Clasificador{ayuda(p['estrategia'])}</dt><dd>{estrategia_txt} · confianza {coma(p.get('confianza'), 2)}{ayuda('confianza')}</dd></div>
  <div><dt>Calibración de referencia{ayuda('calibracion_ref')}</dt><dd>{len(cal.etiquetas)} polímeros · {rango_n} mediciones c/u</dd></div>
  <div><dt>Calibración FLIM{ayuda('irf_flanco')}</dt><dd>{html.escape(calibracion_flim) if calibracion_flim else '—'}</dd></div>
  <div><dt>Escala espacial{ayuda('escala')}</dt><dd>{(coma(escala, 2) + ' µm/píxel') if escala else 'no informada'}</dd></div>
  <div><dt>Fecha de análisis</dt><dd>{fecha}, {hora} h</dd></div>
</dl>"""

    # ================================================================ tablas
    filas_c = []
    for c in CLASES:
        k = int(conteo.get(c, 0))
        if k == 0 and c == NC:
            continue
        lo, hi = wilson(k, n)
        sub = feats[pred == c]
        filas_c.append(
            f"<tr><td>{cod(c)}</td><td class='m'>{NOMBRE[c]}</td>{td(k)}{td(pct(k / n) if n else '—')}"
            f"{td(f'{pct(lo)} – {pct(hi)}' if n else '—', True)}{td(coma(sub.area.sum()))}"
            f"{td(pct(sub.area.sum() / feats.area.sum()) if n else '—')}{td(coma(sub.diametro_eq.median()) if k else '—')}</tr>")
    filas_c.append(f"<tr class='total'><td colspan='2'>Total</td>{td(n)}{td('100,0 %')}{td('')}"
                   f"{td(coma(feats.area.sum()))}{td('100,0 %')}{td(coma(feats.diametro_eq.median()) if n else '—')}</tr>")
    t_comp = tabla(["Polímero", "", "n", "Fracción", f"IC 95 %{ayuda('ic_wilson')}", f"Área ({unidad_area})",
                    "Fracción del área", f"Ø mediano ({unidad_lin})"], filas_c, num={2, 3, 4, 5, 6, 7})

    if a_revisar:
        filas_r = [f"<tr data-id='{lab}' class='fila-roi'>{td(lab)}<td>{cod(feats.loc[lab, 'polimero_predicho'])}</td>"
                   f"{td(coma(feats.loc[lab, 'score_rechazo'], 2))}"
                   f"<td class='m'>{mayus('; '.join(t for k_, t in al[lab] if k_ in ('warn', 'err')))}</td></tr>"
                   for lab in a_revisar]
        t_rev = ("<p>Las partículas siguientes tienen una asignación de menor confiabilidad y conviene verificarlas "
                 "visualmente. Seleccione una fila para ubicarla en las Figuras 1 y 2.</p>"
                 "<div class='bloque'><p class='cap'><b>Tabla 2.</b> Partículas con observaciones.</p>"
                 + tabla(["ID", "Asignación", f"Score{ayuda('score')}", "Observación"], filas_r, num={0, 2},
                         clase="angosta") + "</div>")
    else:
        t_rev = "<p>Ninguna partícula presenta observaciones.</p>"

    enc_cols = {"g_flim": "g FLIM", "s_flim": "s FLIM", "g_esp": "g esp.", "s_esp": "s esp."}
    encabezado = (["ID", "Asignación", f"Score{ayuda('score')}", f"Área ({unidad_area})",
                   f"Ø eq. ({unidad_lin})", f"Aspecto{ayuda('aspecto')}", f"Int. media{ayuda('intensidad')}"]
                  + [enc_cols[c] for c in cols] + (["Real"] if hay_verdad else []) + ["Observaciones"])
    num_idx = {0, 2, 3, 4, 5, 6, *range(7, 7 + len(cols))}

    def fila_p(lab, f):
        celdas = [td(lab), f"<td>{cod(f.polimero_predicho)}</td>", td(coma(f.score_rechazo, 2)), td(coma(f.area)),
                  td(coma(f.diametro_eq)), td(coma(f.relacion_aspecto, 2)), td(coma(f.intensidad_media, 3))]
        celdas += [td(coma(f[c], 3)) for c in cols]
        if hay_verdad:
            celdas.append(f"<td>{cod(f.polimero_real)}</td>")
        celdas.append(f"<td class='m obs'>{mayus('; '.join(t for _, t in al[lab]))}</td>")
        marca = " con-obs" if lab in a_revisar else ""
        return f"<tr data-id='{lab}' data-p='{f.polimero_predicho}' class='fila-roi{marca}'>{''.join(celdas)}</tr>"

    filas_p = "".join(fila_p(lab, f) for lab, f in feats.iterrows())

    csv_cols = ["polimero_predicho", "score_rechazo", "area", "diametro_eq", "area_px", "intensidad_media",
                "intensidad_total", "excentricidad", "solidez", "relacion_aspecto", "centro_fila", "centro_col",
                *cols, *[c for c in ("dispersion_flim", "dispersion_esp", "polimero_real") if c in feats]]
    tabla_csv = feats[csv_cols].copy()
    tabla_csv.insert(0, "muestra", nombre_muestra)
    tabla_csv.insert(0, "informe", id_informe)
    tabla_csv["observaciones"] = ["; ".join(t for _, t in al[l]) for l in feats.index]
    csv = tabla_csv.rename(columns={"area": f"area_{unidad_area}", "diametro_eq": f"diametro_eq_{unidad_lin}"}
                           ).to_csv(index_label="id")

    # ================================================================ imagen interactiva (PNG exacto + SVG)
    from matplotlib.colors import to_rgb
    from skimage.segmentation import find_boundaries
    inten = np.asarray(canales["intensidad"], float)
    lo_i, hi_i = np.nanpercentile(inten, [1, 99.7])
    gris = np.clip((inten - lo_i) / max(hi_i - lo_i, 1e-12), 0, 1) ** 0.85
    rgb = np.repeat(gris[..., None], 3, axis=2)
    labs = resultado.labels
    for lab, f in feats.iterrows():
        m = labs == int(lab)
        col = np.array(to_rgb(COLOR.get(f.polimero_predicho, "#999")))
        rgb[m] = rgb[m] * 0.6 + col * 0.4
        rgb[find_boundaries(m, mode="inner")] = col
    buf = io.BytesIO()
    plt.imsave(buf, np.clip(rgb, 0, 1), format="png")
    img_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    H, W = labs.shape
    fs = max(7.0, W / 38)
    rois_svg = []
    for lab, f in feats.iterrows():
        r = float(np.sqrt(f.area_px / np.pi)) + 2.5
        rois_svg.append(
            f"<g class='roi' data-id='{lab}'><title>Partícula {lab} · {codtxt(f.polimero_predicho)} · score "
            f"{coma(f.score_rechazo, 2)}</title><circle cx='{f.centro_col:.1f}' cy='{f.centro_fila:.1f}' r='{r:.1f}'/>"
            f"<text x='{f.centro_col:.1f}' y='{f.centro_fila:.1f}' font-size='{fs:.1f}'>{lab}</text></g>")
    barra = ""
    if escala:
        ancho_um = W * escala
        largo = next(v for v in (1, 2, 5, 10, 20, 50, 100, 200, 500) if v >= ancho_um / 7)
        px = largo / escala
        x1, y1 = W * 0.96, H * 0.95
        barra = (f"<g class='barra'><rect x='{x1 - px:.1f}' y='{y1:.1f}' width='{px:.1f}' height='{max(2, H / 120):.1f}'/>"
                 f"<text x='{x1 - px / 2:.1f}' y='{y1 - H / 70:.1f}' font-size='{fs * 1.05:.1f}'>{largo} µm</text></g>")
    figura_img = (f"<div class='imgroi'><img src='{img_b64}' alt='Imagen de la muestra con partículas'>"
                  f"<svg viewBox='0 0 {W} {H}' preserveAspectRatio='xMidYMid meet'>{''.join(rois_svg)}{barra}</svg></div>")

    # ================================================================ validación
    validacion = ""
    if hay_verdad and n:
        y, yp = feats.polimero_real.to_numpy(), pred.to_numpy()
        rep = evaluar_clasificacion(y, yp, etiquetas=[c for c in CLASES if c in set(y) | set(yp)])
        rep_p = evaluar_clasificacion(y, yp, incluir_no_clasificable=False)
        seg = resultado.reporte_segmentacion
        f_mc = _a_svg(_fig_confusion(rep.matriz_confusion))
        t_g = tabla(["Indicador", "Valor"], [
            f"<tr><td>Exactitud por partícula{ayuda('exactitud')}</td>{td(coma(rep.exactitud, 3))}</tr>",
            f"<tr><td>F1 macro, solo polímeros{ayuda('f1')}</td>{td(coma(rep_p.f1_macro, 3))}</tr>",
            f"<tr><td>Partículas detectadas / presentes{ayuda('f1_det')}</td>{td(f'{seg.n_emparejadas} / {seg.n_verdaderas}')}</tr>",
            f"<tr><td>IoU medio de contornos{ayuda('iou')}</td>{td(coma(seg.iou_medio, 3))}</tr>"], num={1})
        t_k = tabla(["Clase", f"Precisión{ayuda('precision')}", f"Recall{ayuda('recall')}", f"F1{ayuda('f1')}", "n"],
                    [f"<tr><td>{cod(c)}</td>{td(coma(r.precision, 2))}{td(coma(r.recall, 2))}{td(coma(r.f1, 2))}"
                     f"{td(int(r.soporte))}</tr>" for c, r in rep.por_clase.iterrows()], num={1, 2, 3, 4})
        cap_mc = ("<b>Figura 5.</b> Matriz de confusión" + ayuda("matriz") + ". Filas: composición real; columnas: "
                  "asignación. Porcentaje por fila y, debajo, número de partículas.")
        validacion = f"""
<section><h2><span>3</span>Verificación con composición conocida</h2>
<p>La muestra es de composición conocida, lo que permite verificar el desempeño del procedimiento. Las partículas no
detectadas por la segmentación no intervienen en los indicadores de clasificación.</p>
<div class="dos bloque"><div><p class="cap"><b>Tabla 4.</b> Indicadores globales.</p>{t_g}</div>
<div><p class="cap"><b>Tabla 5.</b> Indicadores por clase.</p>{t_k}</div></div>
<figure class="fig media"><div class="svgwrap">{f_mc}</div><figcaption>{cap_mc}</figcaption></figure>
</section>"""
    s0 = 1 if validacion else 0

    # ================================================================ anexos
    filas_cal = [f"<tr><td>{cod(e)}</td><td class='m'>{NOMBRE[e]}</td>{td(n_cal.get(e, '—'))}"
                 + "".join(td(coma(v, 3)) for v in cal.centroides[e]) + "</tr>" for e in orden_cal]
    t_cal = tabla(["Polímero", "", f"n{ayuda('n_cal')}"] + [enc_cols[c] for c in cal.columnas], filas_cal,
                  num=set(range(2, 3 + len(cal.columnas))))
    from .glosario import GLOSARIO
    claves = ["phasor", "gs", "flim", "espectral", "fusion", "semicirculo", "irf_flanco", "calibracion_ref",
              "confianza", "score", "no_clasif", p["estrategia"], "segmentacion", "watershed", "roi", "dispersion",
              "diametro", "aspecto", "ic_wilson", "huella"]
    if hay_verdad:
        claves += ["exactitud", "precision", "recall", "f1", "matriz", "iou"]
    glosario = "".join(f"<div><dt>{GLOSARIO[k][0]}</dt><dd>{html.escape(GLOSARIO[k][1])}</dd></div>"
                       for k in dict.fromkeys(claves) if k in GLOSARIO)

    # ================================================================ datos para el JS
    elip = {}
    for clave, (cg, cs) in {"flim": ("g_flim", "s_flim"), "esp": ("g_esp", "s_esp")}.items():
        if cg not in cal.columnas:
            continue
        ia, ib = cal.columnas.index(cg), cal.columnas.index(cs)
        lst = []
        for e in orden_cal:
            c0 = np.asarray(cal.centroides[e], float)[[ia, ib]]
            cv = np.asarray(cal.covarianzas[e], float)[np.ix_([ia, ib], [ia, ib])]
            L = np.linalg.cholesky(cv + 1e-12 * np.eye(2))
            t = np.linspace(0, 2 * np.pi, 73)
            n_e = int(n_cal.get(e, 0))
            r2_vis = umbral_hotelling(n_e, 2, conf) if n_e > 3 else float(chi2.ppf(conf, 2))
            pts = c0[:, None] + np.sqrt(r2_vis) * L @ np.vstack([np.cos(t), np.sin(t)])
            lst.append({"p": e, "x": np.round(pts[0], 4).tolist(), "y": np.round(pts[1], 4).tolist()})
        elip[clave] = lst
    rois_json = json.dumps([
        {"id": int(l), "p": f.polimero_predicho, "sc": None if not np.isfinite(f.score_rechazo)
         else round(float(f.score_rechazo), 3), "a": round(float(f.area), 1),
         **{c: None if not np.isfinite(f[c]) else round(float(f[c]), 4) for c in cols}}
        for l, f in feats.iterrows()])
    planos = [k for k in ("flim", "esp") if k in elip]
    paneles = "".join(
        f"<div class='panel'><div class='panel-t'><b>{'ab'[i]}</b> Proyección {'FLIM' if k == 'flim' else 'espectral'}"
        f"{ayuda('semicirculo' if k == 'flim' else 'circulo_espectral')}</div><div id='ph-{k}' class='ph'></div></div>"
        for i, k in enumerate(planos))
    leyenda_ph = "".join(f"<span>{cod(c)}</span>" for c in presentes)
    nombre_pdf = Path(ruta).with_suffix(".pdf").name

    cuerpo = f"""
<header class="cab">
  <div><div class="inst">Universidad Nacional de Entre Ríos · CONICET</div><div class="lab">LAMAE · LaSBI</div></div>
  <div class="num">Informe N.º <b>{id_informe}</b><span>Emitido el {fecha}</span></div>
</header>
<div class="acciones solo-pantalla">
  <a href="{nombre_pdf}" download>Descargar PDF</a><button type="button" onclick="descargarCSV()">Descargar datos (CSV)</button>
</div>

<div class="titulo">
  <div class="tipo">Informe de análisis de microplásticos</div>
  <h1>{html.escape(nombre_muestra)}</h1>
  {f'<p class="desc">{html.escape(descripcion)}</p>' if descripcion else ''}
  <div class="estado est-{est_cls}"><span class="punto"></span>{conclusion}{'<a class="solo-pantalla" href="#observaciones">ver observaciones</a>' if obs else ''}</div>
</div>

<section class="portada">{indicadores}</section>

<section><h2><span>1</span>Muestra y condiciones del análisis</h2>{ficha}</section>

<section><h2><span>2</span>Resultados</h2>
<p class="lead">Se detectaron <b>{n}</b> partículas: <b>{n_mp}</b> se asignaron a alguno de los seis polímeros de
referencia y <b>{n_nc}</b> resultaron no clasificables. {('El polímero mayoritario es ' + cod(dominante) + '.') if len(lideres) == 1 else ''}</p>

<h3>2.1 Composición</h3>
<div class="bloque"><p class="cap"><b>Tabla 1.</b> Partículas y área por polímero asignado. Intervalo de confianza de la fracción según
el método de Wilson.</p>{t_comp}</div>

<h3>2.2 Localización de las partículas{ayuda('segmentacion')}</h3>
<figure class="fig imagen">{figura_img}
<figcaption><b>Figura 1.</b> Imagen de intensidad de Nile Red con las partículas segmentadas, coloreadas según el
polímero asignado. El número es el identificador de las Tablas 2 y 3.<span class="solo-pantalla"> Seleccione una
partícula para ubicarla en la Figura 2 y en la Tabla 3.</span></figcaption></figure>

<h3>2.3 Firma de fluorescencia{ayuda('phasor')}</h3>
<p>Cada punto corresponde a una partícula en el plano de phasores{ayuda('gs')}. Las elipses discontinuas delimitan la
región de aceptación de cada polímero de referencia (confianza {coma(conf, 2)}); las partículas fuera de todas se
informan como no clasificables.</p>
<figure class="fig"><div class="ley-ph">{leyenda_ph}</div><div class="paneles">{paneles}</div>
<figcaption><b>Figura 2.</b> Diagramas de phasores de las partículas sobre las referencias de calibración.
{'Con el clasificador KNN la frontera de decisión no es elíptica; las regiones son orientativas. ' if p['estrategia'] == 'knn' else ''}<span class="solo-pantalla">Pase el cursor para identificar una partícula y selecciónela para resaltarla en la
Figura 1 y en la Tabla 3. Rueda del mouse: zoom; doble clic: vista completa.</span></figcaption></figure>

<h3>2.4 Tamaño y confiabilidad de la asignación</h3>
<div class="dos">
<figure class="fig"><div class="svgwrap">{f_tam}</div><figcaption><b>Figura 3.</b> Diámetro equivalente por polímero; la línea vertical indica la mediana.</figcaption></figure>
<figure class="fig"><div class="svgwrap">{f_conf}</div><figcaption><b>Figura 4.</b> Distribución del score de rechazo{ayuda('score')}. Valores mayores a 1 corresponden a partículas no clasificables.</figcaption></figure>
</div>

<h3>2.5 Partículas con observaciones{ayuda('revisar')}</h3>
{t_rev}

<h3>2.6 Resultados por partícula</h3>
<div class="filtros solo-pantalla">
  <input id="buscar" type="search" placeholder="Buscar ID u observación" aria-label="Buscar">
  <select id="filtro" aria-label="Filtrar por polímero"><option value="">Todos los polímeros</option>
  {''.join(f'<option value="{c}">{codtxt(c)}</option>' for c in presentes)}</select>
  <label><input type="checkbox" id="solo_obs"> Solo con observaciones</label>
</div>
<p class="cap"><b>Tabla 3.</b> Resultados individuales. Score: distancia al polímero asignado relativa al umbral de
aceptación. Aspecto: cociente entre los ejes mayor y menor.</p>
<div class="desborde">{tabla(encabezado, [filas_p], num=num_idx, clase="chica", id_="tabla")}</div>
</section>
{validacion}
<section id="observaciones"><h2><span>{3 + s0}</span>Observaciones</h2>
<p>Resultado del análisis: <span class="estado en-linea est-{est_cls}"><span class="punto"></span>{conclusion}</span></p>
{f'<ol class="lista">{"".join(f"<li>{html.escape(t)}</li>" for t in obs)}</ol>' if obs else '<p>Sin observaciones.</p>'}
</section>

<section><h2><span>{4 + s0}</span>Método</h2>
<p>La imagen de intensidad se segmentó por umbral de Otsu, con cierre morfológico y separación de partículas en
contacto mediante watershed{ayuda('watershed')}. Para cada partícula se calculó la mediana de los phasores de sus
píxeles (phasorpy, <span class="mono">phasor_center</span>). Las partículas se clasificaron con el método de
{estrategia_txt} contra la calibración de referencia (Anexo A), con un criterio de rechazo «no clasificable» para un
nivel de confianza de {coma(p.get('confianza'), 2)}.</p>
</section>

<section><h2><span>{5 + s0}</span>Alcance y limitaciones</h2>
<ol class="lista">
<li>La clasificación se limita a los seis polímeros de la calibración (PET, HDPE, PVC, LDPE, PP y PS); un polímero no
incluido se informa como no clasificable.</li>
<li>«No clasificable» no implica ausencia de plástico: comprende materia orgánica, autofluorescencia, mezclas y
polímeros ajenos a la calibración.</li>
<li>La asignación es válida si la calibración se adquirió con el mismo equipo, configuración y protocolo de
envejecimiento que la muestra.</li>
<li>No se detectan partículas menores al tamaño mínimo de segmentación; los tamaños corresponden a la proyección 2D.</li>
</ol>
<div class="firmas"><div>Analizó</div><div>Revisó</div><div>Aprobó</div></div>
</section>

<section class="anexo"><h2><span>A</span>Calibración de referencia</h2>
<div class="bloque"><p class="cap"><b>Tabla A.1.</b> Número de mediciones y centroide de cada polímero en el plano de phasores.</p>
{t_cal}</div>
<p class="nota">{html.escape(str(cal.metadatos)) if cal.metadatos else 'La calibración no registra metadatos de adquisición.'}</p>
<p class="nota">Archivo analizado: {html.escape(Path(archivo).name) if archivo else '—'} · SHA-256{ayuda('huella')}
<span class="mono">{huella or '—'}</span> · napari-mp-classifier {version_paquete} · Python {platform.python_version()}</p>
</section>

<section class="anexo"><h2><span>B</span>Glosario</h2><dl class="glosario">{glosario}</dl></section>
"""

    css = (CSS_AYUDA + CSS_INFORME).replace("__ID__", id_informe)
    doc = ("<!doctype html><html lang='es'><head><meta charset='utf-8'>"
           "<meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>{id_informe} · {html.escape(nombre_muestra)}</title>"
           "<link rel='preconnect' href='https://fonts.googleapis.com'>"
           "<link href='https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,500;8..60,600;8..60,700"
           "&family=Source+Sans+3:ital,wght@0,400;0,600;0,700;1,400&family=IBM+Plex+Mono:wght@400;500&display=swap' "
           "rel='stylesheet'>"
           "<script src='https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js'></script>"
           f"<style>{css}</style></head><body><main>{cuerpo}</main>"
           f"<script>const ROIS={rois_json};const COL={json.dumps(COLOR)};const CSV={json.dumps(csv)};"
           f"const ARCHIVO_CSV={json.dumps(id_informe + '_particulas.csv')};const ELIP={json.dumps(elip)};"
           f"const PLANOS={json.dumps(planos)};</script>"
           f"<script>{JS_INFORME}</script><script>{JS_AYUDA}</script></body></html>")
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(doc, encoding="utf-8")
    if exportar_pdf:
        exportar_a_pdf(ruta)
    return ruta


def exportar_a_pdf(ruta_html: str | Path) -> Path | None:
    """Imprime un informe HTML a PDF con Chrome/Chromium en modo headless.

    Usa ``--no-pdf-header-footer``: el PDF no lleva la ruta del archivo ni la fecha del
    navegador; el pie de página lo define el propio informe (``@page``).

    Parameters
    ----------
    ruta_html : str or pathlib.Path
        Informe HTML.

    Returns
    -------
    pathlib.Path or None
        Ruta del PDF, o ``None`` si no hay Chrome/Chromium, la impresión falló o la
        variable de entorno ``NAPARI_MP_SIN_PDF`` está definida (la usan los tests).
    """
    import os

    if os.environ.get("NAPARI_MP_SIN_PDF"):
        return None
    import shutil
    import subprocess

    chrome = next((shutil.which(c) for c in ("google-chrome", "chromium", "chromium-browser") if shutil.which(c)), None)
    if chrome is None:
        return None
    destino = Path(ruta_html).with_suffix(".pdf")
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    "--virtual-time-budget=10000", f"--print-to-pdf={destino}", Path(ruta_html).resolve().as_uri()],
                   check=False, capture_output=True, timeout=120)
    return destino if destino.exists() else None


CSS_INFORME = """
:root { color-scheme: light; --ink:#1a1d23; --ink-2:#4e5361; --ink-3:#868b98; --rule:#d9dce3; --rule-2:#eceef2;
  --acc:#1f3a5f; --acc-2:#eef2f8; --ok:#1e7a4c; --warn:#a46a00; --err:#b0302f; }
* { box-sizing:border-box; }
html, body { margin:0; background:#fff; color:var(--ink); }
body { font:15px/1.6 'Source Sans 3', 'Segoe UI', Arial, sans-serif; -webkit-font-smoothing:antialiased;
  -webkit-print-color-adjust:exact; print-color-adjust:exact; }
main { max-width:900px; margin:0 auto; padding:0 28px 64px; }
a { color:var(--acc); }
.mono { font-family:'IBM Plex Mono', monospace; font-size:.86em; word-break:break-all; }

/* encabezado */
.cab { display:flex; justify-content:space-between; align-items:flex-end; gap:16px; padding:22px 0 12px;
  border-top:5px solid var(--acc); border-bottom:1px solid var(--rule); }
.cab .inst { font-size:11.5px; font-weight:700; letter-spacing:.14em; text-transform:uppercase; color:var(--acc); }
.cab .lab { font-size:13px; color:var(--ink-2); }
.cab .num { text-align:right; font-size:13px; color:var(--ink-2); line-height:1.45; }
.cab .num b { color:var(--ink); font-family:'IBM Plex Mono', monospace; font-weight:500; }
.cab .num span { display:block; }
.acciones { display:flex; justify-content:flex-end; gap:8px; margin-top:10px; }
.acciones a, .acciones button { font:600 13px 'Source Sans 3', sans-serif; color:var(--acc); background:#fff;
  border:1px solid #c9d3e2; border-radius:3px; padding:5px 12px; cursor:pointer; text-decoration:none; }
.acciones a:hover, .acciones button:hover { background:var(--acc-2); }

/* título */
.titulo { margin:28px 0 18px; }
.tipo { font-size:12px; font-weight:700; letter-spacing:.14em; text-transform:uppercase; color:var(--ink-3); }
h1 { font:600 34px/1.15 'Source Serif 4', Georgia, serif; margin:6px 0 8px; letter-spacing:-.01em; }
.desc { margin:0 0 12px; color:var(--ink-2); font-size:16px; max-width:70ch; }
.estado { display:inline-flex; align-items:center; gap:8px; font-weight:700; font-size:14px; padding:5px 12px 5px 10px;
  border:1px solid currentColor; border-radius:3px; }
.estado .punto { width:9px; height:9px; border-radius:50%; background:currentColor; }
.estado a { font-weight:400; font-size:13px; margin-left:6px; color:inherit; }
.estado.en-linea { border:0; padding:0; }
.est-ok { color:var(--ok); } .est-warn { color:var(--warn); } .est-err { color:var(--err); }

/* portada */
.portada { margin:6px 0 10px; padding:20px 0 16px; border-top:1px solid var(--rule); border-bottom:1px solid var(--rule); }
.kpis { display:grid; grid-template-columns:repeat(5, 1fr); }
.kpis > div { padding:0 16px; border-left:1px solid var(--rule-2); display:flex; flex-direction:column; gap:2px; }
.kpis > div:first-child { padding-left:0; border-left:0; }
.kpis .v { font:600 32px/1.1 'Source Serif 4', Georgia, serif; font-variant-numeric:lining-nums tabular-nums; }
.kpis .v small { font:600 14px 'Source Sans 3', sans-serif; color:var(--ink-3); }
.kpis .l { font-size:12.5px; color:var(--ink-2); }
.comp { display:flex; height:30px; margin:20px 0 8px; border-radius:2px; overflow:hidden; gap:2px; background:#fff; }
.seg-c { display:flex; align-items:center; justify-content:center; min-width:4px; color:#fff; font-size:12px;
  text-shadow:0 0 3px rgba(0,0,0,.45); }
.comp-ley { display:flex; flex-wrap:wrap; gap:6px 18px; font-size:13px; color:var(--ink-2); }
.comp-ley em { font-style:normal; color:var(--ink-3); }

/* secciones */
section { margin-top:38px; }
h2 { font:600 22px/1.25 'Source Serif 4', Georgia, serif; margin:0 0 14px; display:flex; align-items:baseline; gap:12px; }
h2 > span:first-child { font:700 15px 'Source Sans 3', sans-serif; color:var(--acc); min-width:18px; }
h3 { font:700 15.5px/1.3 'Source Sans 3', sans-serif; margin:28px 0 8px; color:var(--ink); }
p { margin:8px 0; max-width:78ch; }
.lead { font-size:16px; }
.cap { margin:14px 0 6px; font-size:13.5px; color:var(--ink-2); max-width:none; }
.cap b, figcaption b { color:var(--acc); font-weight:700; }
.nota { font-size:13px; color:var(--ink-3); max-width:none; }
.pol { display:inline-flex; align-items:center; gap:6px; white-space:nowrap; font-weight:600; color:var(--ink); }
.pol i { width:10px; height:10px; display:inline-block; border-radius:2px; box-shadow:inset 0 0 0 .5px rgba(0,0,0,.3); }

/* ficha */
.ficha { display:grid; grid-template-columns:repeat(3, 1fr); gap:0; margin:0; border-top:1px solid var(--rule); }
.ficha > div { padding:10px 14px 10px 0; border-bottom:1px solid var(--rule-2); }
.ficha > div.ancho { grid-column:1 / -1; }
.ficha dt { font-size:11.5px; font-weight:700; letter-spacing:.06em; text-transform:uppercase; color:var(--ink-3); }
.ficha dd { margin:3px 0 0; font-size:14.5px; }

/* tablas */
table { border-collapse:collapse; width:100%; font-size:14px; margin:2px 0 6px; }
th, td { padding:7px 10px; text-align:left; vertical-align:middle; }
th { font-size:11.5px; font-weight:700; letter-spacing:.04em; text-transform:uppercase; color:var(--ink-2);
  border-bottom:1.5px solid var(--ink); white-space:nowrap; background:#fff; }
td { border-bottom:1px solid var(--rule-2); }
td.r, th.r { text-align:right; font-variant-numeric:tabular-nums lining-nums; white-space:nowrap; }
td.m { color:var(--ink-2); }
tr.total td { border-top:1.5px solid var(--ink); border-bottom:0; font-weight:700; }
tbody tr:last-child td { border-bottom:1.5px solid var(--ink); }
tr.total + tr td, tbody tr.total:last-child td { border-bottom:0; }
table.angosta { width:auto; min-width:62%; }
table.chica { font-size:13px; } table.chica th, table.chica td { padding:6px 8px; }
.desborde { overflow:auto; max-height:560px; border-bottom:1.5px solid var(--ink); }
.desborde thead th { position:sticky; top:0; z-index:1; }
.desborde tbody tr:last-child td { border-bottom:0; }
td.obs { font-size:12.5px; }
tr.fila-roi { cursor:pointer; }
tr.fila-roi:hover td { background:#f5f8fc; }
tr.marcada td { background:#fff2c2 !important; }
.filtros { display:flex; flex-wrap:wrap; gap:10px 14px; align-items:center; margin:6px 0 2px; font-size:13.5px; color:var(--ink-2); }
.filtros input[type=search], .filtros select { font:14px 'Source Sans 3', sans-serif; padding:6px 10px;
  border:1px solid var(--rule); border-radius:3px; background:#fff; color:var(--ink); }
.filtros input[type=search] { min-width:230px; }
#tabla th { cursor:pointer; }

/* figuras */
figure.fig { margin:12px 0 6px; break-inside:avoid; }
figcaption { font-size:13.5px; color:var(--ink-2); margin-top:8px; line-height:1.5; }
.svgwrap svg { width:100%; height:auto; display:block; }
figure.media .svgwrap { max-width:520px; }
.dos { display:grid; grid-template-columns:1fr 1fr; gap:26px; align-items:start; }
.imgroi { position:relative; max-width:600px; margin:0 auto; line-height:0; }
.imgroi img { width:100%; height:auto; display:block; image-rendering:auto; border-radius:2px; }
.imgroi svg { position:absolute; inset:0; width:100%; height:100%; }
.roi circle { fill:transparent; stroke:transparent; stroke-width:1.5; cursor:pointer; vector-effect:non-scaling-stroke; }
.roi:hover circle { stroke:#fff; stroke-dasharray:3 2; }
.roi.marcada circle { stroke:#ffd23f; stroke-width:3; }
.roi text { fill:#fff; font-family:'Source Sans 3', sans-serif; font-weight:700; text-anchor:middle; dominant-baseline:central;
  paint-order:stroke; stroke:rgba(0,0,0,.75); stroke-width:2px; pointer-events:none; }
.barra rect { fill:#fff; } .barra text { fill:#fff; font-family:'Source Sans 3', sans-serif; font-weight:700; text-anchor:middle;
  paint-order:stroke; stroke:rgba(0,0,0,.6); stroke-width:2px; }
.ley-ph { display:flex; flex-wrap:wrap; gap:6px 18px; font-size:13px; margin-bottom:6px; }
.paneles { display:grid; grid-template-columns:1fr 1fr; gap:18px; }
.panel-t { font-size:13px; color:var(--ink-2); margin-bottom:2px; }
.panel-t b { color:var(--ink); margin-right:4px; }
.ph { width:100%; aspect-ratio:1 / 1; }

/* listas, firmas, glosario */
ol.lista { margin:6px 0; padding-left:22px; max-width:78ch; } ol.lista li { margin:4px 0; }
.firmas { display:grid; grid-template-columns:repeat(3, 1fr); gap:36px; margin-top:56px; }
.firmas div { border-top:1px solid var(--ink); padding-top:6px; font-size:12.5px; color:var(--ink-2); text-align:center; }
dl.glosario { display:grid; grid-template-columns:1fr 1fr; gap:4px 34px; margin:0; font-size:13.5px; }
dl.glosario div { break-inside:avoid; padding:8px 0; border-bottom:1px solid var(--rule-2); }
dl.glosario dt { font-weight:700; } dl.glosario dd { margin:2px 0 0; color:var(--ink-2); }

/* ayudas: círculo discreto */
.ayuda { width:15px; height:15px; margin-left:5px; border:1px solid #b9bdc7; background:#fff; color:#6b7080;
  font:700 9.5px/1 'Source Sans 3', sans-serif; vertical-align:2px; }
.ayuda:hover, .ayuda:focus-visible, .ayuda[aria-expanded="true"] { background:var(--acc); border-color:var(--acc); color:#fff; }
#globo-ayuda { background:#fff; color:var(--ink); border:1px solid var(--rule); border-radius:4px;
  box-shadow:0 6px 24px rgba(20,30,50,.14); font:13.5px/1.5 'Source Sans 3', sans-serif; }
#globo-ayuda b { color:var(--acc); }

@media screen and (max-width: 680px) {
  main { padding:0 16px 48px; }
  .kpis { grid-template-columns:repeat(2, 1fr); row-gap:14px; } .kpis > div { border-left:0; padding-left:0; }
  .ficha, .dos, .paneles, dl.glosario { grid-template-columns:1fr; }
  .cab { flex-direction:column; align-items:flex-start; } .cab .num { text-align:left; }
  table.angosta { width:100%; }
}

/* impresión: lo mismo que en pantalla, sin controles ni encabezados del navegador */
@page {
  size:A4; margin:14mm 13mm 16mm;
  @bottom-left { content:"Informe __ID__"; font:9px 'Source Sans 3', Arial, sans-serif; color:#868b98; }
  @bottom-right { content:"Página " counter(page) " de " counter(pages); font:9px 'Source Sans 3', Arial, sans-serif; color:#868b98; }
}
@media print {
  body { font-size:12.5px; }
  main { max-width:none; padding:0; }
  .solo-pantalla, .filtros, .ayuda, .modebar-container { display:none !important; }
  .cab { padding-top:10px; }
  h1 { font-size:28px; }
  .kpis .v { font-size:26px; }
  section { margin-top:26px; }
  h2, h3 { break-after:avoid; }
  figure, table.angosta, .portada, .ficha, .firmas, tr, .bloque { break-inside:avoid; }
  .cap { break-after:avoid; }
  .desborde { max-height:none; overflow:visible; border-bottom:0; }
  .desborde tbody tr:last-child td { border-bottom:1.5px solid var(--ink); }
  .desborde thead th { position:static; }
  .anexo { break-before:page; }
  .imgroi { max-width:440px; }
  .paneles { grid-template-columns:1fr 1fr; gap:14px; }
  .ph { aspect-ratio:auto; width:100%; height:320px; }
  .panel, .paneles { break-inside:avoid; }
  .dos { gap:18px; }
  tr.marcada td { background:transparent !important; }
  .roi.marcada circle { stroke:transparent; }
}
"""

JS_INFORME = r"""
function descargarCSV() { const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([CSV], { type:'text/csv' })); a.download = ARCHIVO_CSV; a.click(); }
let MARCA = null;
const ORDEN = ['PET', 'HDPE', 'PVC', 'LDPE', 'PP', 'PS', 'no_clasificable'];
function dibujar(proy) {
  const div = document.getElementById('ph-' + proy);
  if (!window.Plotly || !div) return;
  const gx = proy === 'flim' ? 'g_flim' : 'g_esp', sx = proy === 'flim' ? 's_flim' : 's_esp';
  const trazas = (ELIP[proy] || []).map(e => ({ type:'scatter', mode:'lines', x:e.x, y:e.y, fill:'toself',
    fillcolor: COL[e.p] + '16', line:{ color: COL[e.p], width:1.1, dash:'dash' }, hoverinfo:'skip', showlegend:false }));
  ORDEN.filter(c => ROIS.some(r => r.p === c)).forEach(c => { const rs = ROIS.filter(r => r.p === c && r[gx] !== null);
    trazas.push({ type:'scatter', mode:'markers', name: c === 'no_clasificable' ? 'No clasif.' : c,
      x: rs.map(r => r[gx]), y: rs.map(r => r[sx]),
      marker: { size: rs.map(r => 7 + Math.sqrt(r.a) / 2.2), color: COL[c], opacity:.95,
        symbol: c === 'no_clasificable' ? 'circle-open' : 'circle', line:{ width: c === 'no_clasificable' ? 1.4 : .8, color: c === 'no_clasificable' ? COL[c] : '#fff' } },
      customdata: rs.map(r => [r.id, r.sc === null ? '—' : r.sc.toFixed(2).replace('.', ','), String(r.a).replace('.', ',')]),
      hovertemplate: '<b>Partícula %{customdata[0]}</b> · %{fullData.name}<br>score %{customdata[1]} · área %{customdata[2]}<extra></extra>', showlegend:false }); });
  const tops = (ELIP[proy] || []).map(e => { const i = e.y.indexOf(Math.max(...e.y)); return [e.p, e.x[i], e.y[i]]; });
  trazas.push({ type:'scatter', mode:'text', x: tops.map(t => t[1]), y: tops.map(t => t[2]), textposition:'top center',
    text: tops.map(t => '<b>' + t[0] + '</b>'), textfont:{ size:11, color:'#1a1d23' }, hoverinfo:'skip', showlegend:false });
  const m = MARCA !== null ? ROIS.find(r => r.id === MARCA && r[gx] !== null) : null;
  if (m) trazas.push({ type:'scatter', mode:'markers+text', x:[m[gx]], y:[m[sx]], text:['  ' + m.id], textposition:'middle right',
    textfont:{ size:12, color:'#1a1d23' }, marker:{ size:24, color:'rgba(0,0,0,0)', line:{ width:2.4, color:'#1a1d23' } }, hoverinfo:'skip', showlegend:false });
  const xs = ROIS.map(r => r[gx]).filter(v => v !== null).concat((ELIP[proy] || []).flatMap(e => e.x));
  const ys = ROIS.map(r => r[sx]).filter(v => v !== null).concat((ELIP[proy] || []).flatMap(e => e.y));
  const cx = (Math.max(...xs) + Math.min(...xs)) / 2, cy = (Math.max(...ys) + Math.min(...ys)) / 2;
  const lado = 1.14 * Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys), 0.1);
  const ref = proy === 'flim'
    ? [{ type:'path', path:'M 0 0 ' + Array.from({ length:121 }, (_, i) => { const t = Math.PI * i / 120; return `L ${0.5 - 0.5 * Math.cos(t)} ${0.5 * Math.sin(t)}`; }).join(' '), line:{ color:'#9aa0ab', width:1 }, layer:'below' }]
    : [{ type:'circle', x0:-1, y0:-1, x1:1, y1:1, line:{ color:'#9aa0ab', width:1 }, layer:'below' }];
  const eje = t => ({ title:{ text:t, font:{ size:12 } }, gridcolor:'#f0f1f4', zeroline:false, linecolor:'#1a1d23', mirror:true,
    ticks:'inside', ticklen:4, tickfont:{ size:10.5 } });
  Plotly.react(div, trazas, { margin:{ l:48, r:10, t:6, b:40 }, paper_bgcolor:'#fff', plot_bgcolor:'#fff',
    font:{ family:'Source Sans 3, Arial, sans-serif', color:'#4e5361' }, shapes: ref, hovermode:'closest', separators:', ',
    xaxis: Object.assign(eje('g'), { range:[cx - lado / 2, cx + lado / 2] }),
    yaxis: Object.assign(eje('s'), { range:[cy - lado / 2, cy + lado / 2], scaleanchor:'x' }) },
    { displaylogo:false, responsive:true, scrollZoom:true, displayModeBar:'hover',
      modeBarButtonsToRemove:['lasso2d', 'select2d', 'autoScale2d', 'toImage'] });
  if (!div.dataset.escucha) { div.on('plotly_click', ev => { const cd = ev.points[0].customdata; if (cd) marcar(cd[0], 'diagrama'); }); div.dataset.escucha = '1'; }
}
function dibujarTodo() { PLANOS.forEach(dibujar); }
function marcar(id, origen) {
  MARCA = (MARCA === id && origen !== 'diagrama') ? null : id;
  dibujarTodo();
  document.querySelectorAll('.marcada').forEach(x => x.classList.remove('marcada'));
  if (MARCA === null) return;
  document.querySelectorAll(`[data-id="${id}"]`).forEach(x => x.classList.add('marcada'));
  if (origen !== 'tabla') { const tr = document.querySelector(`#tabla tr[data-id="${id}"]`); if (tr) tr.scrollIntoView({ block:'nearest', behavior:'smooth' }); }
}
dibujarTodo();
document.querySelectorAll('tr.fila-roi').forEach(tr => tr.addEventListener('click', () => marcar(Number(tr.dataset.id), 'tabla')));
document.querySelectorAll('.roi').forEach(g => g.addEventListener('click', () => marcar(Number(g.dataset.id), 'imagen')));
function filtrar() { const q = document.getElementById('buscar').value.toLowerCase(), f = document.getElementById('filtro').value,
  so = document.getElementById('solo_obs').checked;
  document.querySelectorAll('#tabla tbody tr').forEach(tr => {
    const ok = (!q || tr.innerText.toLowerCase().includes(q)) && (!f || tr.dataset.p === f) && (!so || tr.classList.contains('con-obs'));
    tr.style.display = ok ? '' : 'none'; }); }
['buscar', 'filtro', 'solo_obs'].forEach(i => { const el = document.getElementById(i); if (el) el.addEventListener('input', filtrar); });
document.querySelectorAll('#tabla th').forEach((th, i) => th.addEventListener('click', e => {
  if (e.target.closest('.ayuda')) return;
  const tb = document.querySelector('#tabla tbody'), filas = [...tb.rows], asc = th.dataset.asc !== '1'; th.dataset.asc = asc ? '1' : '0';
  const v = r => { const s = r.cells[i].innerText.trim().replace(/ /g, '').replace(',', '.'); const n = parseFloat(s); return isNaN(n) ? s : n; };
  filas.sort((a, b) => (v(a) > v(b) ? 1 : v(a) < v(b) ? -1 : 0) * (asc ? 1 : -1)); filas.forEach(f => tb.appendChild(f)); }));
function ajustar() { if (!window.Plotly) return; PLANOS.forEach(k => { const d = document.getElementById('ph-' + k);
  if (d && d.clientWidth) Plotly.relayout(d, { width: d.clientWidth, height: d.clientHeight }); }); }
function antesDeImprimir() { MARCA = null; document.querySelectorAll('.marcada').forEach(x => x.classList.remove('marcada')); dibujarTodo(); ajustar(); }
window.addEventListener('beforeprint', antesDeImprimir);
window.matchMedia('print').addEventListener('change', e => { if (e.matches) antesDeImprimir(); else { PLANOS.forEach(k => { const d = document.getElementById('ph-' + k);
  if (d) Plotly.relayout(d, { width: null, height: null, autosize: true }); }); } });
"""
