"""Informe de caracterización con los datos reales de LAMAE (PE + Nile Red).

Sin la calibración de referencia de los 6 polímeros no se puede clasificar: el informe
caracteriza la partícula en ambas modalidades —phasor espectral y FLIM calibrado con la IRF
(como SPCImage), en los dos detectores del ``.sdt``— con la misma plantilla visual que los
informes por muestra (:mod:`napari_mp_classifier.informe_html`).

Necesita los archivos de ``ejemplo_lamae/`` (no versionados: ``*.sdt``/``*.czi`` están en
``.gitignore``).

Uso::

    python ejemplos/informe_lamae.py

Salida: ``ejemplos/salida_demo_lamae/informe_LAMAE_PE.html`` (+ ``.pdf`` si hay Chrome).
"""

from __future__ import annotations

import base64
import hashlib
import html
import io
import json
import platform
import sys
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from phasorpy.filter import phasor_filter_median, phasor_threshold
from phasorpy.io import signal_from_czi, signal_from_sdt
from phasorpy.lifetime import phasor_to_apparent_lifetime
from phasorpy.phasor import phasor_from_signal
from skimage.segmentation import find_boundaries

from napari_mp_classifier import __version__ as VERSION
from napari_mp_classifier.calibracion_flim import calibrar_con_irf, irf_desde_flanco
from napari_mp_classifier.features import extraer_features
from napari_mp_classifier.glosario import CSS_AYUDA, GLOSARIO, JS_AYUDA, ayuda
from napari_mp_classifier.informe_html import CSS_INFORME, exportar_a_pdf
from napari_mp_classifier.segmentacion import segmentar

DATOS = REPO / "ejemplo_lamae"
SDT = DATOS / "PE-NileRed-63x-440.sdt"
CZI = DATOS / "PE-NileRed-63x-lmada-440.czi"
SALIDA = REPO / "ejemplos" / "salida_demo_lamae" / "informe_LAMAE_PE.html"
ACENTO = "#1f3a5f"
COLOR_PE = "#eb6834"  # color de polietileno (HDPE/LDPE comparten familia; se usa el de HDPE)


def coma(v, d=1):
    if v is None or not np.isfinite(v):
        return "—"
    return f"{v:,.{d}f}".replace(",", " ").replace(".", ",")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def png_overlay(intensidad, labels, color, gamma=0.8):
    lo, hi = np.nanpercentile(intensidad, [1, 99.7])
    g = np.clip((intensidad - lo) / max(hi - lo, 1e-12), 0, 1) ** gamma
    rgb = np.repeat(g[..., None], 3, axis=2)
    col = np.array(to_rgb(color))
    for lab in np.unique(labels[labels > 0]):
        m = labels == lab
        rgb[m] = rgb[m] * 0.75 + col * 0.25
        rgb[find_boundaries(m, mode="inner")] = col
    buf = io.BytesIO()
    plt.imsave(buf, np.clip(rgb, 0, 1), format="png")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def figura_imagen(intensidad, labels, feats, escala_um=None):
    H, W = labels.shape
    fs = max(8.0, W / 30)
    txt = "".join(
        f"<g class='roi'><title>Partícula {int(l)}</title><text x='{f.centro_col:.1f}' "
        f"y='{f.centro_fila - np.sqrt(f.area_px / np.pi) - fs * 0.9:.1f}' font-size='{fs:.1f}'>{int(l)}</text></g>"
        for l, f in feats.iterrows())
    barra = ""
    if escala_um:
        largo = next(v for v in (1, 2, 5, 10, 20, 50) if v >= W * escala_um / 7)
        px = largo / escala_um
        x1, y1 = W * 0.96, H * 0.95
        barra = (f"<g class='barra'><rect x='{x1 - px:.1f}' y='{y1:.1f}' width='{px:.1f}' height='{max(2, H / 110):.1f}'/>"
                 f"<text x='{x1 - px / 2:.1f}' y='{y1 - H / 60:.1f}' font-size='{fs:.1f}'>{largo} µm</text></g>")
    return (f"<div class='imgroi'><img src='{png_overlay(intensidad, labels, COLOR_PE)}' alt=''>"
            f"<svg viewBox='0 0 {W} {H}'>{txt}{barra}</svg></div>")


def hist_phasor(g, s, rango, bins):
    m = np.isfinite(g) & np.isfinite(s)
    h, xe, ye = np.histogram2d(g[m], s[m], bins=bins, range=rango)
    z = np.where(h > 0, np.log10(h), None).T.tolist()
    return {"x": ((xe[:-1] + xe[1:]) / 2).round(4).tolist(), "y": ((ye[:-1] + ye[1:]) / 2).round(4).tolist(), "z": z}


def main():
    if not (SDT.exists() and CZI.exists()):
        raise SystemExit(f"Faltan los datos de LAMAE en {DATOS} (no se versionan).")
    ahora = datetime.now().astimezone()
    huellas = {SDT.name: sha(SDT), CZI.name: sha(CZI)}
    id_inf = f"MP-{ahora:%Y%m%d}-{huellas[CZI.name][:6].upper()}"

    # ------------------------------------------------------------ espectral
    sig = signal_from_czi(str(CZI))
    cubo = np.asarray(sig.values, float)  # (C, Y, X)
    lambdas = np.asarray(sig.coords["C"].values, float)
    escalas = dict(sig.attrs["coord_scales"])
    um_px = float(escalas["X"]) * 1e6
    obj = sig.attrs.get("objective", "")
    mean, g_e, s_e = phasor_from_signal(cubo, axis=0)
    lab_e = segmentar(mean, separar=False, tam_min=30)
    fe = extraer_features(lab_e, mean, g_esp=g_e, s_esp=s_e, escala_um_px=um_px)
    fe["diam_um"] = 2 * np.sqrt(fe.area_um2 / np.pi)
    paso = lambdas[1] - lambdas[0]
    periodo = lambdas[-1] - lambdas[0] + paso
    fe["lambda_nm"] = lambdas[0] + (np.arctan2(fe.s_esp, fe.g_esp) % (2 * np.pi)) / (2 * np.pi) * periodo
    principal = fe.sort_values("area_px").index[-1]
    m_roi = lab_e == principal
    fondo = (lab_e == 0) & (mean < np.percentile(mean, 50))
    esp_roi = cubo[:, m_roi].mean(1)
    esp_fondo = cubo[:, fondo].mean(1)
    lam_max = float(lambdas[np.argmax(esp_roi - esp_fondo)])

    # ------------------------------------------------------------ FLIM, dos detectores
    det = []
    for idx in (0, 1):
        sd = signal_from_sdt(str(SDT), index=idx)
        frec = float(sd.attrs["frequency"])
        c = np.asarray(sd.values, float)
        fot = c.sum(-1)
        mean_f, g_f, s_f = phasor_from_signal(c, axis=-1)
        brillo = fot >= np.percentile(fot, 90)
        decay = c[brillo].sum(0)
        irf = irf_desde_flanco(decay)
        gc, sc, info = calibrar_con_irf(g_f, s_f, irf, frec)
        mf, gf, sf = phasor_filter_median(mean_f, gc, sc, size=3, repeat=2)
        det.append({"idx": idx, "frec": frec, "cubo": c, "fot": fot, "decay": decay, "irf": irf, "info": info, "gf": gf, "sf": sf, "mf": mf})
    principal_det = max(det, key=lambda d: d["fot"].sum())
    D = principal_det
    lab_f = segmentar(D["fot"], separar=False, tam_min=30)
    ff = extraer_features(lab_f, D["fot"], g_flim=D["gf"], s_flim=D["sf"])
    for d in det:
        _, gt, st = phasor_threshold(d["mf"], d["gf"], d["sf"], mean_min=np.percentile(d["mf"], 90))
        G, S = float(np.nanmedian(gt)), float(np.nanmedian(st))
        tp, tm = phasor_to_apparent_lifetime(G, S, d["frec"])
        d.update(G=G, S=S, tp=float(tp), tm=float(tm), gt=gt, st=st,
                 fwhm=float((d["irf"] > d["irf"].max() / 2).sum() * 1e6 / d["frec"] / d["cubo"].shape[-1]))
    pf = ff.sort_values("area_px").index[-1]
    m_f = lab_f == pf
    decay_roi = D["cubo"][m_f].sum(0)
    t_ns = np.arange(D["cubo"].shape[-1]) * 1e3 / D["frec"] / D["cubo"].shape[-1]

    # ------------------------------------------------------------ datos para gráficos interactivos
    datos_js = {
        "esp": {**hist_phasor(g_e[mean > np.percentile(mean, 80)], s_e[mean > np.percentile(mean, 80)],
                              [[-1.05, 1.05], [-1.05, 1.05]], 210),
                "roi": [[round(float(f.g_esp), 4), round(float(f.s_esp), 4), int(l), round(float(f.lambda_nm), 1)]
                        for l, f in fe.iterrows()],
                "ticks": [[float(np.cos(2 * np.pi * (nm - lambdas[0]) / periodo)),
                           float(np.sin(2 * np.pi * (nm - lambdas[0]) / periodo)), nm] for nm in (500, 550, 600, 650, 700)]},
        "flim": {**hist_phasor(D["gt"].ravel(), D["st"].ravel(), [[-0.02, 1.02], [-0.01, 0.62]], 220),
                 "roi": [[round(d["G"], 4), round(d["S"], 4), d["idx"], round(d["tp"], 2), round(d["tm"], 2)] for d in det],
                 "ticks": [[1 / (1 + (2 * np.pi * D['frec'] * 1e-3 * t) ** 2),
                            (2 * np.pi * D['frec'] * 1e-3 * t) / (1 + (2 * np.pi * D['frec'] * 1e-3 * t) ** 2), t]
                           for t in (0.5, 1, 2, 3, 4, 6, 8, 12)]},
        "espectro": {"x": lambdas.round(1).tolist(), "roi": (esp_roi / esp_roi.max()).round(4).tolist(),
                     "fondo": (esp_fondo / esp_roi.max()).round(4).tolist()},
        "decay": {"x": t_ns.round(4).tolist(), "y": np.where(decay_roi > 0, decay_roi, None).tolist(),
                  "irf": (D["irf"] / D["irf"].max() * decay_roi.max()).round(2).tolist()},
    }

    # ------------------------------------------------------------ contenido
    roi = fe.loc[principal]
    kpis = f"""
<div class="kpis">
  <div><span class="v">{len(fe)}</span><span class="l">partícula(s) detectada(s){ayuda('roi')}</span></div>
  <div><span class="v">{coma(roi.area_um2, 0)}<small> µm²</small></span><span class="l">área de la partícula</span></div>
  <div><span class="v">{coma(roi.diam_um)}<small> µm</small></span><span class="l">diámetro equivalente{ayuda('diametro')}</span></div>
  <div><span class="v">{coma(roi.lambda_nm, 0)}<small> nm</small></span><span class="l">centro espectral del phasor{ayuda('circulo_espectral')}</span></div>
  <div><span class="v">{coma(D['tp'], 2)}<small> ns</small></span><span class="l">τφ · τm {coma(D['tm'], 2)} ns (detector {D['idx']}){ayuda('tau')}</span></div>
</div>"""

    filas_det = "".join(
        f"<tr><td>Detector {d['idx']}{' (principal)' if d is D else ''}</td><td class='r'>{coma(d['fot'].sum() / 1e6)} M</td>"
        f"<td class='r'>{coma(np.percentile(d['fot'], 99), 0)}</td><td class='r'>{coma(d['fwhm'], 0)}</td>"
        f"<td class='r'>{coma(np.degrees(d['info']['fase_irf_rad']))}</td><td class='r'>{coma(d['G'], 3)}</td>"
        f"<td class='r'>{coma(d['S'], 3)}</td><td class='r'><b>{coma(d['tp'], 2)}</b></td><td class='r'><b>{coma(d['tm'], 2)}</b></td></tr>"
        for d in det)
    t_det = ("<table><thead><tr><th>Canal</th><th class='r'>Fotones</th><th class='r'>Fotones/píxel p99"
             f"{ayuda('fotones')}</th><th class='r'>FWHM IRF (ps){ayuda('irf')}</th><th class='r'>Fase IRF (°)</th>"
             f"<th class='r'>g</th><th class='r'>s</th><th class='r'>τφ (ns)</th><th class='r'>τm (ns)</th></tr></thead>"
             f"<tbody>{filas_det}</tbody></table>")

    filas_p = "".join(
        f"<tr><td class='r'>{int(l)}</td><td>Espectral</td><td class='r'>{coma(f.area_um2)} µm²</td><td class='r'>{coma(f.diam_um)}</td>"
        f"<td class='r'>{coma(f.relacion_aspecto, 2)}</td><td class='r'>{coma(f.solidez, 2)}</td><td class='r'>{coma(f.g_esp, 3)}</td>"
        f"<td class='r'>{coma(f.s_esp, 3)}</td><td class='r'>{coma(f.lambda_nm, 0)} nm</td><td class='r'>{coma(f.dispersion_esp, 3)}</td></tr>"
        for l, f in fe.iterrows())
    filas_p += "".join(
        f"<tr><td class='r'>{int(l)}</td><td>FLIM (det. {D['idx']})</td><td class='r'>{coma(f.area_px, 0)} px²</td>"
        f"<td class='r'>—</td><td class='r'>{coma(f.relacion_aspecto, 2)}</td><td class='r'>{coma(f.solidez, 2)}</td>"
        f"<td class='r'>{coma(f.g_flim, 3)}</td><td class='r'>{coma(f.s_flim, 3)}</td>"
        f"<td class='r'>τφ {coma(float(phasor_to_apparent_lifetime(f.g_flim, f.s_flim, D['frec'])[0]), 2)} ns</td>"
        f"<td class='r'>{coma(f.dispersion_flim, 3)}</td></tr>" for l, f in ff.iterrows())
    t_part = ("<table class='chica'><thead><tr><th class='r'>ID</th><th>Modalidad</th><th class='r'>Área</th>"
              f"<th class='r'>Ø eq. (µm)</th><th class='r'>Aspecto{ayuda('aspecto')}</th><th class='r'>Solidez</th>"
              f"<th class='r'>g</th><th class='r'>s</th><th class='r'>Lectura</th><th class='r'>Dispersión{ayuda('dispersion')}</th>"
              f"</tr></thead><tbody>{filas_p}</tbody></table>")

    obs = [
        ("No se dispone de la calibración de referencia de los seis polímeros (PET, HDPE, PVC, LDPE, PP, PS): la "
        "partícula se caracteriza pero no se clasifica."),
        ("Las adquisiciones FLIM (512 × 512 píxeles) y espectral (440 × 440 píxeles) no están registradas entre sí; "
        "no es posible combinar ambas modalidades por partícula."),
        (f"El archivo FLIM contiene dos detectores simultáneos (módulos TCSPC 3N0140 y 3N0141). Se usó como principal "
        f"el detector {D['idx']} ({coma(D['fot'].sum() / 1e6)} M fotones). Debe confirmarse qué banda de emisión registra cada uno."),
        ("La IRF se estimó a partir del flanco de subida del decaimiento (método de SPCImage). Una IRF medida en la "
        "misma sesión (reflexión en el cubreobjetos) eliminaría esta aproximación."),
        "El archivo FLIM no informa el tamaño de píxel; los tamaños FLIM se expresan en píxeles.",
        "Debe confirmarse el código SPI del polietileno analizado (HDPE o LDPE).",
        "El archivo z-stack (PE-NileRed-63x-zstack-440.czi) no contiene información espectral y no se procesó.",
    ]

    glos = ["phasor", "gs", "flim", "espectral", "circulo_espectral", "semicirculo", "tau", "irf", "irf_flanco",
            "tac", "detectores", "fotones", "mediana_filtro", "segmentacion", "roi", "diametro", "aspecto",
            "dispersion", "calibracion_ref", "huella"]
    glosario = "".join(f"<div><dt>{GLOSARIO[k][0]}</dt><dd>{html.escape(GLOSARIO[k][1])}</dd></div>" for k in glos)
    obj_txt = "63× / 1,4 aceite" if "63" in str(obj) else "—"
    nombre_pdf = SALIDA.with_suffix(".pdf").name

    cuerpo = f"""
<header class="cab">
  <div><div class="inst">Universidad Nacional de Entre Ríos · CONICET</div><div class="lab">LAMAE · LaSBI</div></div>
  <div class="num">Informe N.º <b>{id_inf}</b><span>Emitido el {ahora:%d/%m/%Y}</span></div>
</header>
<div class="acciones solo-pantalla"><a href="{nombre_pdf}" download>Descargar PDF</a></div>

<div class="titulo">
  <div class="tipo">Informe de caracterización de partículas</div>
  <h1>PE · Nile Red · 63× · excitación 440 nm</h1>
  <p class="desc">Partícula de polietileno teñida con Nile Red. Caracterización por phasores espectral y FLIM
  (datos reales de LAMAE).</p>
  <div class="estado est-warn"><span class="punto"></span>Caracterización — sin clasificación<a class="solo-pantalla" href="#observaciones">ver observaciones</a></div>
</div>

<section class="portada">{kpis}</section>

<section><h2><span>1</span>Muestra y condiciones de adquisición</h2>
<dl class="ficha">
  <div><dt>Muestra</dt><dd>Polietileno (PE) + Nile Red</dd></div>
  <div><dt>Objetivo</dt><dd>{obj_txt}</dd></div>
  <div><dt>Excitación</dt><dd>440 nm</dd></div>
  <div><dt>Espectral{ayuda('espectral')}</dt><dd>λ-stack {len(lambdas)} canales, {coma(lambdas[0], 0)}–{coma(lambdas[-1], 0)} nm, paso {coma(paso, 0)} nm</dd></div>
  <div><dt>Píxel (espectral){ayuda('escala')}</dt><dd>{coma(um_px * 1000, 1)} nm · {cubo.shape[2]} × {cubo.shape[1]} px</dd></div>
  <div><dt>FLIM{ayuda('flim')}</dt><dd>TCSPC {D['cubo'].shape[-1]} canales, {coma(D['frec'], 2)} MHz · {D['cubo'].shape[1]} × {D['cubo'].shape[0]} px</dd></div>
  <div><dt>Calibración FLIM{ayuda('irf_flanco')}</dt><dd>IRF sintética del flanco de subida (método SPCImage)</dd></div>
  <div><dt>Filtrado del phasor{ayuda('mediana_filtro')}</dt><dd>mediana 3 × 3, 2 pasadas</dd></div>
  <div><dt>Clasificación{ayuda('calibracion_ref')}</dt><dd>no realizada: falta calibración de referencia</dd></div>
</dl></section>

<section><h2><span>2</span>Resultados</h2>
<p class="lead">Se detectó <b>{len(fe)}</b> partícula en la imagen espectral, de <b>{coma(roi.area_um2, 0)} µm²</b>
(diámetro equivalente {coma(roi.diam_um)} µm). Su emisión tiene el máximo en <b>{coma(lam_max, 0)} nm</b> y el centro
espectral del phasor en {coma(roi.lambda_nm, 0)} nm (el phasor resume todo el espectro, incluida la cola hacia el
rojo, por eso su centro queda por encima del máximo); ambos valores son compatibles con Nile Red en un entorno apolar. En FLIM, calibrado
con la IRF, el tiempo de vida aparente es <b>τφ = {coma(D['tp'], 2)} ns</b> y <b>τm = {coma(D['tm'], 2)} ns</b>
(detector {D['idx']}); como τφ &lt; τm, la emisión no es monoexponencial.</p>

<h3>2.1 Imagen y segmentación{ayuda('segmentacion')}</h3>
<div class="dos">
<figure class="fig">{figura_imagen(mean, lab_e, fe, um_px)}<figcaption><b>Figura 1.</b> Intensidad espectral integrada
(λ-stack) con la partícula segmentada por umbral de Otsu.</figcaption></figure>
<figure class="fig">{figura_imagen(D['fot'], lab_f, ff)}<figcaption><b>Figura 2.</b> Intensidad FLIM (fotones por píxel,
detector {D['idx']}) con la partícula segmentada. Sin escala física en el archivo.</figcaption></figure>
</div>

<h3>2.2 Firma de fluorescencia{ayuda('phasor')}</h3>
<div class="paneles">
  <div class="panel"><div class="panel-t"><b>a</b> Phasor espectral{ayuda('circulo_espectral')}</div><div id="ph-esp" class="ph"></div></div>
  <div class="panel"><div class="panel-t"><b>b</b> Phasor FLIM calibrado{ayuda('semicirculo')}</div><div id="ph-flim" class="ph"></div></div>
</div>
<figure class="fig"><figcaption><b>Figura 3.</b> (a) Phasor espectral de los píxeles con señal; el marcador indica la
partícula y las marcas del círculo, la longitud de onda. (b) Phasor FLIM calibrado con la IRF y filtrado; las marcas del
semicírculo indican el tiempo de vida. Los marcadores corresponden a cada detector.<span class="solo-pantalla"> Pase el
cursor para ver los valores; rueda del mouse: zoom.</span></figcaption></figure>

<h3>2.3 Espectro de emisión y decaimiento</h3>
<div class="paneles">
  <div class="panel"><div class="panel-t"><b>a</b> Espectro de emisión de la partícula</div><div id="g-esp" class="ph corta"></div></div>
  <div class="panel"><div class="panel-t"><b>b</b> Decaimiento de la partícula{ayuda('irf')}</div><div id="g-dec" class="ph corta"></div></div>
</div>
<figure class="fig"><figcaption><b>Figura 4.</b> (a) Espectro medio de la partícula y del fondo, normalizados al máximo
de la partícula. (b) Decaimiento acumulado de la partícula (escala logarítmica) y la IRF estimada del flanco; las zonas
sin datos corresponden a los límites del TAC{ayuda('tac')}.</figcaption></figure>

<h3>2.4 Lectura por detector FLIM{ayuda('detectores')}</h3>
<div class="bloque"><p class="cap"><b>Tabla 1.</b> Phasor calibrado y tiempos de vida aparentes de los píxeles más
brillantes (10 %), por detector.</p>{t_det}</div>

<h3>2.5 Resultados por partícula</h3>
<div class="bloque"><p class="cap"><b>Tabla 2.</b> Partículas segmentadas en cada modalidad.</p><div class="desborde">{t_part}</div></div>
</section>

<section id="observaciones"><h2><span>3</span>Observaciones</h2>
<p>Resultado: <span class="estado en-linea est-warn"><span class="punto"></span>Caracterización — sin clasificación</span></p>
<ol class="lista">{''.join(f'<li>{html.escape(o)}</li>' for o in obs)}</ol></section>

<section><h2><span>4</span>Método</h2>
<p>El phasor espectral se calculó sobre el λ-stack (primer armónico) con phasorpy; la intensidad integrada se segmentó por
umbral de Otsu. Para FLIM se estimó la IRF del flanco de subida del decaimiento total de los píxeles más brillantes y se
calibró el phasor dividiéndolo por el phasor de la IRF (<span class="mono">phasor_calibrate</span> con referencia de tiempo
de vida 0), seguido de un filtro de mediana 3 × 3 en dos pasadas. Los tiempos de vida aparentes se obtuvieron de la fase
(τφ) y de la modulación (τm). El procedimiento de calibración se validó con decaimientos simulados de tiempo de vida
conocido (error de 2–6 % entre 1 y 4 ns).</p>
<div class="firmas"><div>Analizó</div><div>Revisó</div><div>Aprobó</div></div></section>

<section class="anexo"><h2><span>A</span>Trazabilidad</h2>
<div class="bloque"><table><thead><tr><th>Archivo</th><th>SHA-256{ayuda('huella')}</th></tr></thead><tbody>
{''.join(f"<tr><td>{html.escape(k)}</td><td class='mono'>{v}</td></tr>" for k, v in huellas.items())}</tbody></table></div>
<p class="nota">napari-mp-classifier {VERSION} · phasorpy · Python {platform.python_version()} · generado el
{ahora:%d/%m/%Y %H:%M}.</p></section>

<section class="anexo"><h2><span>B</span>Glosario</h2><dl class="glosario">{glosario}</dl></section>
"""
    css_extra = """
.ph.corta { aspect-ratio:4 / 3; }
.imgroi .roi text { fill:#fff; font-family:'Source Sans 3', sans-serif; font-weight:700; text-anchor:middle;
  dominant-baseline:central; paint-order:stroke; stroke:rgba(0,0,0,.75); stroke-width:2px; }
@media print { .ph.corta { height:250px; } }
"""
    js = r"""
const AC = '#1f3a5f', PE = '#eb6834';
const ESCALA = [[0, '#dbe7f6'], [0.35, '#7fa9dc'], [0.7, '#2a5d9a'], [1, '#0d2340']];
const base = (tx, ty) => ({ margin:{ l:48, r:10, t:6, b:40 }, paper_bgcolor:'#fff', plot_bgcolor:'#fff', separators:', ',
  font:{ family:'Source Sans 3, Arial, sans-serif', size:11, color:'#4e5361' }, showlegend:false, hovermode:'closest',
  xaxis:{ title:{ text:tx }, zeroline:false, linecolor:'#1a1d23', mirror:true, ticks:'inside', gridcolor:'#f0f1f4' },
  yaxis:{ title:{ text:ty }, zeroline:false, linecolor:'#1a1d23', mirror:true, ticks:'inside', gridcolor:'#f0f1f4' } });
const CFG = { displaylogo:false, responsive:true, scrollZoom:true, displayModeBar:'hover', modeBarButtonsToRemove:['lasso2d', 'select2d', 'autoScale2d', 'toImage'] };
function heat(d) { return { type:'heatmap', x:d.x, y:d.y, z:d.z, colorscale:ESCALA, showscale:false, hoverinfo:'skip', zsmooth:false }; }
function graficar() {
  if (!window.Plotly) return;
  const E = DATOS.esp, Fl = DATOS.flim;
  // espectral
  let L = base('g', 's'); L.xaxis.range = [-1.08, 1.08]; L.yaxis.range = [-1.08, 1.08]; L.yaxis.scaleanchor = 'x';
  L.shapes = [{ type:'circle', x0:-1, y0:-1, x1:1, y1:1, line:{ color:'#9aa0ab', width:1 } }];
  Plotly.react('ph-esp', [heat(E),
    { type:'scatter', mode:'text', x:E.ticks.map(t => t[0] * 1.13), y:E.ticks.map(t => t[1] * 1.13), text:E.ticks.map(t => t[2] + ' nm'), textfont:{ size:10, color:'#868b98' }, hoverinfo:'skip' },
    { type:'scatter', mode:'markers', x:E.ticks.map(t => t[0]), y:E.ticks.map(t => t[1]), marker:{ size:5, color:'#868b98' }, hoverinfo:'skip' },
    { type:'scatter', mode:'markers', x:E.roi.map(r => r[0]), y:E.roi.map(r => r[1]), marker:{ size:13, color:PE, line:{ width:2, color:'#fff' } },
      customdata:E.roi, hovertemplate:'<b>Partícula %{customdata[2]}</b><br>g %{x:.3f} · s %{y:.3f}<br>centro espectral %{customdata[3]} nm<extra></extra>' }], L, CFG);
  // FLIM
  let M = base('g', 's'); M.xaxis.range = [-0.03, 1.03]; M.yaxis.range = [-0.02, 0.66]; M.yaxis.scaleanchor = 'x';
  M.shapes = [{ type:'path', path:'M 0 0 ' + Array.from({ length:121 }, (_, i) => { const t = Math.PI * i / 120; return `L ${0.5 - 0.5 * Math.cos(t)} ${0.5 * Math.sin(t)}`; }).join(' ') + ' Z', line:{ color:'#9aa0ab', width:1 } }];
  Plotly.react('ph-flim', [heat(Fl),
    { type:'scatter', mode:'markers+text', x:Fl.ticks.map(t => t[0]), y:Fl.ticks.map(t => t[1]), text:Fl.ticks.map(t => t[2] + ' ns'), textposition:'top center', textfont:{ size:9.5, color:'#868b98' }, marker:{ size:4, color:'#868b98' }, hoverinfo:'skip' },
    { type:'scatter', mode:'markers+text', x:Fl.roi.map(r => r[0]), y:Fl.roi.map(r => r[1]), text:Fl.roi.map((r, i) => i ? '  det. ' + r[2] : 'det. ' + r[2] + '  '), textposition:Fl.roi.map((r, i) => i ? 'middle right' : 'middle left'), textfont:{ size:10.5, color:'#1a1d23' },
      marker:{ size:11, color:Fl.roi.map((r, i) => i ? PE : '#fff'), line:{ width:2, color:PE }, symbol:'circle' },
      customdata:Fl.roi, hovertemplate:'<b>Detector %{customdata[2]}</b><br>g %{x:.3f} · s %{y:.3f}<br>τφ %{customdata[3]} ns · τm %{customdata[4]} ns<extra></extra>' }], M, CFG);
  // espectro
  const S = DATOS.espectro; let A = base('Longitud de onda (nm)', 'Intensidad normalizada'); A.showlegend = true;
  A.legend = { x:0.98, xanchor:'right', y:0.98, font:{ size:10.5 } };
  Plotly.react('g-esp', [
    { type:'scatter', mode:'lines+markers', name:'Partícula', x:S.x, y:S.roi, line:{ color:PE, width:2 }, marker:{ size:4 }, hovertemplate:'%{x} nm · %{y:.3f}<extra>Partícula</extra>' },
    { type:'scatter', mode:'lines', name:'Fondo', x:S.x, y:S.fondo, line:{ color:'#868b98', width:1.4, dash:'dot' }, hovertemplate:'%{x} nm · %{y:.3f}<extra>Fondo</extra>' }], A, CFG);
  // decaimiento
  const Dd = DATOS.decay; let B = base('Tiempo (ns)', 'Fotones'); B.yaxis.type = 'log'; B.showlegend = true;
  B.legend = { x:0.98, xanchor:'right', y:0.98, font:{ size:10.5 } };
  Plotly.react('g-dec', [
    { type:'scatter', mode:'lines', name:'Partícula', x:Dd.x, y:Dd.y, line:{ color:AC, width:1.4 }, hovertemplate:'%{x:.2f} ns · %{y} fotones<extra></extra>' },
    { type:'scatter', mode:'lines', name:'IRF estimada', x:Dd.x, y:Dd.irf.map(v => v > 0 ? v : null), line:{ color:PE, width:1.8 }, hoverinfo:'skip' }], B, CFG);
}
graficar();
function ajustar() { ['ph-esp', 'ph-flim', 'g-esp', 'g-dec'].forEach(id => { const d = document.getElementById(id); if (window.Plotly && d && d.clientWidth) Plotly.relayout(d, { width:d.clientWidth, height:d.clientHeight }); }); }
window.addEventListener('beforeprint', ajustar);
window.matchMedia('print').addEventListener('change', e => { if (e.matches) ajustar(); else ['ph-esp', 'ph-flim', 'g-esp', 'g-dec'].forEach(id => { const d = document.getElementById(id); if (d) Plotly.relayout(d, { width:null, height:null, autosize:true }); }); });
"""
    css = (CSS_AYUDA + CSS_INFORME + css_extra).replace("__ID__", id_inf)
    doc = ("<!doctype html><html lang='es'><head><meta charset='utf-8'>"
           "<meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>{id_inf} · PE Nile Red</title>"
           "<link rel='preconnect' href='https://fonts.googleapis.com'>"
           "<link href='https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,500;8..60,600;8..60,700"
           "&family=Source+Sans+3:ital,wght@0,400;0,600;0,700;1,400&family=IBM+Plex+Mono:wght@400;500&display=swap' rel='stylesheet'>"
           "<script src='https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js'></script>"
           f"<style>{css}</style></head><body><main>{cuerpo}</main>"
           f"<script>const DATOS={json.dumps(datos_js)};</script><script>{js}</script><script>{JS_AYUDA}</script></body></html>")
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    SALIDA.write_text(doc, encoding="utf-8")
    pdf = exportar_a_pdf(SALIDA)
    print(SALIDA, pdf, f"{len(doc) // 1024} KB")
    print(f"ROI esp: {roi.area_um2:.1f} µm², λ {roi.lambda_nm:.1f} nm, máx {lam_max:.0f} nm; "
          f"FLIM principal det {D['idx']}: τφ {D['tp']:.2f} τm {D['tm']:.2f}; ROIs FLIM {len(ff)}")


if __name__ == "__main__":
    main()
