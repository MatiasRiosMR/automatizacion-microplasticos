"""Generación de datos sintéticos de phasores para tests y desarrollo (Fase 1).

No pretende reproducir valores físicos reales: genera 6 clusters separables con dispersión
y solapamiento realistas, más una población de "materia orgánica / autofluorescencia" que
debe caer como ``"no_clasificable"``. Los centroides FLIM se derivan de tiempos de vida
plausibles mediante :func:`phasorpy.lifetime.phasor_from_lifetime`; los espectrales se
colocan a mano sobre el arco del plano de phasores.

Cuando el equipo entregue ``.sdt``/``.czi`` reales, estos generadores se reemplazan por
la calibración medida; los tests de ``clasificador`` y ``metricas`` siguen valiendo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from phasorpy.lifetime import phasor_from_lifetime

from napari_mp_classifier import POLIMEROS

# Tiempos de vida (ns) ILUSTRATIVOS de Nile Red en cada polímero, ordenados como POLIMEROS.
# Elegidos solo para dar 6 clusters FLIM separables; NO son valores medidos.
LIFETIMES_ILUSTRATIVOS_NS: dict[str, float] = {
    "PET": 2.1,
    "HDPE": 3.9,
    "PVC": 1.6,
    "LDPE": 3.4,
    "PP": 4.6,
    "PS": 2.8,
}

# Posiciones espectrales ILUSTRATIVAS (g_esp, s_esp) sobre el arco de phasores.
POSICIONES_ESPECTRALES_ILUSTRATIVAS: dict[str, tuple[float, float]] = {
    "PET": (0.55, 0.50),
    "HDPE": (0.10, 0.30),
    "PVC": (0.70, 0.44),
    "LDPE": (0.20, 0.40),
    "PP": (0.02, 0.15),
    "PS": (0.40, 0.49),
}

FRECUENCIA_FLIM_MHZ = 80.0


def centroides_referencia(modalidad: str = "flim") -> dict[str, np.ndarray]:
    """Centroides teóricos de los 6 polímeros para la modalidad pedida.

    Parameters
    ----------
    modalidad : {"flim", "espectral", "fusion"}

    Returns
    -------
    dict[str, numpy.ndarray]
        Centroide por polímero: 2D para ``flim``/``espectral``, 4D para ``fusion``
        (orden: ``g_flim, s_flim, g_esp, s_esp``).
    """
    g_flim, s_flim = phasor_from_lifetime(
        FRECUENCIA_FLIM_MHZ, [LIFETIMES_ILUSTRATIVOS_NS[p] for p in POLIMEROS]
    )
    flim = {p: np.array([g_flim[i], s_flim[i]]) for i, p in enumerate(POLIMEROS)}
    esp = {p: np.array(POSICIONES_ESPECTRALES_ILUSTRATIVAS[p]) for p in POLIMEROS}

    if modalidad == "flim":
        return flim
    if modalidad == "espectral":
        return esp
    if modalidad == "fusion":
        return {p: np.concatenate([flim[p], esp[p]]) for p in POLIMEROS}
    raise ValueError(f"modalidad desconocida: {modalidad!r}")


def _cov_isotropica(d: int, sigma: float) -> np.ndarray:
    return np.eye(d) * sigma**2


def generar_calibracion(
    modalidad: str = "flim",
    n_por_polimero: int = 60,
    sigma: float = 0.02,
    semilla: int = 0,
) -> pd.DataFrame:
    """Genera mediciones de calibración sintéticas (una fila por medición).

    Parameters
    ----------
    modalidad : {"flim", "espectral", "fusion"}
    n_por_polimero : int
        Mediciones simuladas por polímero.
    sigma : float
        Desvío estándar del ruido gaussiano isotrópico en el plano de phasores.
    semilla : int

    Returns
    -------
    pandas.DataFrame
        Columnas según modalidad + columna ``polimero``.
    """
    rng = np.random.default_rng(semilla)
    centroides = centroides_referencia(modalidad)
    columnas = _columnas(modalidad)
    d = len(columnas)

    filas = []
    for p, centro in centroides.items():
        muestras = rng.multivariate_normal(centro, _cov_isotropica(d, sigma), size=n_por_polimero)
        for fila in muestras:
            filas.append({**dict(zip(columnas, fila)), "polimero": p})
    return pd.DataFrame(filas)


def _deriva_envejecimiento(centro: np.ndarray, centro_global: np.ndarray) -> np.ndarray:
    """Dirección unitaria de deriva por envejecimiento: del cluster hacia el centro común.

    Modelo sintético: al degradarse la matriz polimérica (abrasión + H2O2 + UV) la
    respuesta de Nile Red pierde especificidad y los 6 clusters **convergen** hacia una
    firma común. Es el escenario que hace más difícil clasificar, y el que interesa medir
    para la robustez (los 6 polímeros de referencia ya se calibran envejecidos con el
    mismo protocolo; el riesgo es la *variabilidad del grado* de envejecimiento).
    """
    direccion = centro_global - centro
    norma = np.linalg.norm(direccion)
    return direccion / norma if norma > 0 else np.zeros_like(direccion)


def generar_particulas(
    modalidad: str = "flim",
    n_por_polimero: int = 40,
    n_no_clasificables: int = 60,
    sigma: float = 0.025,
    grado_envejecimiento: float = 0.0,
    semilla: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Genera partículas a clasificar con verdad de terreno.

    Incluye ``n_no_clasificables`` puntos de "materia orgánica / autofluorescencia":
    una nube ancha y desplazada del conjunto de clusters, etiquetada ``"no_clasificable"``.

    Parameters
    ----------
    modalidad : {"flim", "espectral", "fusion"}
    n_por_polimero : int
    n_no_clasificables : int
    sigma : float
        Desvío base del ruido en el plano de phasores.
    grado_envejecimiento : float, optional
        Desajuste del estado de envejecimiento de la muestra respecto del estándar de
        calibración. ``0`` = igual que la calibración (los 6 clusters de referencia ya son
        de polímero envejecido con abrasión + H2O2 [+ UV]). Un valor ``g > 0`` desplaza
        cada cluster una fracción ``g`` hacia la firma común (convergencia por degradación)
        e infla el ruido en ``1 + 0.6·|g|``; ``g < 0`` simula muestra **menos** meteorizada
        que el estándar (deriva en sentido opuesto). Por defecto ``0``.
    semilla : int

    Returns
    -------
    X : numpy.ndarray, shape (n, d)
    y : numpy.ndarray of str, shape (n,)
    """
    rng = np.random.default_rng(semilla)
    centroides = centroides_referencia(modalidad)
    columnas = _columnas(modalidad)
    d = len(columnas)
    centro_global = np.mean(list(centroides.values()), axis=0)

    # Escala típica de separación entre clusters, para dar unidades a `grado`.
    escala = float(np.mean([
        np.linalg.norm(c - centro_global) for c in centroides.values()
    ]))
    sigma_efectivo = sigma * (1.0 + 0.6 * abs(grado_envejecimiento))

    X_list, y_list = [], []
    for p, centro in centroides.items():
        centro_muestra = centro + grado_envejecimiento * escala * _deriva_envejecimiento(
            centro, centro_global
        )
        X_list.append(
            rng.multivariate_normal(centro_muestra, _cov_isotropica(d, sigma_efectivo),
                                    size=n_por_polimero)
        )
        y_list.append(np.full(n_por_polimero, p))

    if n_no_clasificables > 0:
        centro_organico = centro_global + 0.22
        cov_organico = _cov_isotropica(d, sigma * 4)
        X_list.append(rng.multivariate_normal(centro_organico, cov_organico, size=n_no_clasificables))
        y_list.append(np.full(n_no_clasificables, "no_clasificable"))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    orden = rng.permutation(len(X))
    return X[orden], y[orden]


def _columnas(modalidad: str) -> list[str]:
    return {
        "flim": ["g_flim", "s_flim"],
        "espectral": ["g_esp", "s_esp"],
        "fusion": ["g_flim", "s_flim", "g_esp", "s_esp"],
    }[modalidad]


# ---------------------------------------------------------------------------
# Fase 2 — imágenes sintéticas de "muestra" (blobs + fondo + materia orgánica)
# ---------------------------------------------------------------------------

CANALES_IMAGEN: tuple[str, ...] = ("intensidad", "g_flim", "s_flim", "g_esp", "s_esp")


def generar_imagen_muestra(
    forma: tuple[int, int] = (320, 320),
    n_por_polimero: int = 4,
    n_materia_organica: int = 8,
    n_pares_contacto: int = 4,
    radio_px: tuple[float, float] = (6.0, 10.0),
    sigma_phasor: float = 0.03,
    fondo: float = 0.03,
    sigma_fondo: float = 0.01,
    grado_envejecimiento: float = 0.0,
    semilla: int = 0,
) -> tuple[dict[str, np.ndarray], dict]:
    """Genera una imagen sintética de muestra multimodal para probar la Fase 2.

    Simula un campo de microscopía de Nile Red con partículas de microplástico
    (``n_por_polimero`` por cada uno de los 6 polímeros), cuerpos de "materia orgánica /
    autofluorescencia" (más grandes, más tenues, con firma de phasor difusa y desplazada)
    y un fondo ruidoso. Las partículas se colocan con una separación mínima; además se
    agregan ``n_pares_contacto`` partículas pegadas a otra ya colocada, para ejercitar la
    separación por ``watershed``.

    Cada píxel con señal lleva coordenadas de phasor FLIM y espectral: el centroide del
    polímero (o el de la nube orgánica) más ruido gaussiano. Donde varias partículas se
    solapan, el phasor es el promedio ponderado por intensidad (imita el mezclado real y
    genera dispersión intra-ROI en el borde de contacto). El fondo tiene phasor ``NaN``.

    Parameters
    ----------
    forma : tuple of int
        Alto y ancho de la imagen en px.
    n_por_polimero : int
        Partículas de microplástico por polímero.
    n_materia_organica : int
        Cuerpos de materia orgánica / autofluorescencia (verdad ``"no_clasificable"``).
    n_pares_contacto : int
        Partículas extra pegadas a otra ya colocada (generan pares en contacto).
    radio_px : tuple of float
        Rango del radio de las partículas de polímero (los cuerpos orgánicos son ~1,6×
        más grandes). El perfil es súper-gaussiano (borde más marcado que una gaussiana).
    sigma_phasor : float
        Desvío del ruido de phasor por píxel en las partículas de polímero
        (3× en los cuerpos orgánicos).
    fondo : float
        Intensidad media del fondo.
    sigma_fondo : float
        Ruido del fondo.
    grado_envejecimiento : float, optional
        Desajuste del estado de envejecimiento respecto del estándar de calibración
        (ver :func:`generar_particulas`). Por defecto ``0``.
    semilla : int

    Returns
    -------
    canales : dict[str, numpy.ndarray]
        ``"intensidad"`` (2D, ``>= 0``) y ``"g_flim"``, ``"s_flim"``, ``"g_esp"``,
        ``"s_esp"`` (2D, con ``NaN`` en el fondo).
    verdad : dict
        - ``"labels"`` : numpy.ndarray 2D int — segmentación de verdad de terreno
          (0 = fondo; partículas en contacto llevan labels distintos).
        - ``"polimero"`` : dict[int, str] — polímero (o ``"no_clasificable"``) de cada label.
    """
    from skimage.segmentation import relabel_sequential

    rng = np.random.default_rng(semilla)
    alto, ancho = forma
    yy, xx = np.mgrid[0:alto, 0:ancho].astype(float)
    margen = int(np.ceil(radio_px[1] * 2))
    sep_min = 2.3 * radio_px[1]

    centros_ref = centroides_referencia("fusion")
    centro_global = np.mean(list(centros_ref.values()), axis=0)
    escala = float(np.mean([np.linalg.norm(c - centro_global) for c in centros_ref.values()]))
    centros_fusion = {
        p: c + grado_envejecimiento * escala * _deriva_envejecimiento(c, centro_global)
        for p, c in centros_ref.items()
    }
    centro_organico = centro_global + 0.22

    plan: list[tuple[str, np.ndarray, float]] = []
    for p in POLIMEROS:
        plan += [("polimero", centros_fusion[p], 1.0) for _ in range(n_por_polimero)]
    plan += [("organico", centro_organico, 3.0) for _ in range(n_materia_organica)]
    rng.shuffle(plan)

    # Centros de las partículas "sueltas": rechazo por distancia mínima.
    posiciones: list[tuple[float, float]] = []
    radios: list[float] = []
    for tipo, _, _ in plan:
        radio_base = rng.uniform(*radio_px) * (1.6 if tipo == "organico" else 1.0)
        for _ in range(200):
            cy, cx = rng.uniform(margen, alto - margen), rng.uniform(margen, ancho - margen)
            if all((cy - py) ** 2 + (cx - px) ** 2 > sep_min**2 for py, px in posiciones):
                break
        posiciones.append((cy, cx))
        radios.append(radio_base)

    # Partículas de contacto: pegadas a una partícula de polímero ya colocada.
    indices_polimero = [i for i, (t, _, _) in enumerate(plan) if t == "polimero"]
    for _ in range(n_pares_contacto):
        base = int(rng.choice(indices_polimero))
        py, px = posiciones[base]
        radio = radios[base]
        angulo = rng.uniform(0, 2 * np.pi)
        distancia = radio * rng.uniform(1.3, 1.6)
        posiciones.append((py + distancia * np.sin(angulo), px + distancia * np.cos(angulo)))
        radios.append(radio * rng.uniform(0.9, 1.1))
        plan.append(plan[base])

    intensidad = np.abs(rng.normal(fondo, sigma_fondo, forma))
    acumulador = {c: np.zeros(forma) for c in CANALES_IMAGEN[1:]}
    peso = np.zeros(forma)
    contribuciones: list[np.ndarray] = []
    codigos: list[str] = []

    ruido_envejecimiento = 1.0 + 0.6 * abs(grado_envejecimiento)
    for (cy, cx), radio, (tipo, vector_phasor, factor_ruido) in zip(posiciones, radios, plan):
        amplitud = rng.uniform(0.25, 0.4) if tipo == "organico" else rng.uniform(0.7, 1.0)
        r2 = (yy - cy) ** 2 + (xx - cx) ** 2
        contrib = amplitud * np.exp(-((r2 / (2.0 * radio**2)) ** 2))  # súper-gaussiana
        intensidad += contrib
        escala_ruido = sigma_phasor * factor_ruido * (ruido_envejecimiento if tipo == "polimero" else 1.0)
        for i, canal in enumerate(CANALES_IMAGEN[1:]):
            ruido = rng.normal(0.0, escala_ruido, forma)
            acumulador[canal] += contrib * (vector_phasor[i] + ruido)
        peso += contrib
        contribuciones.append(contrib)
        codigos.append(
            "no_clasificable" if tipo == "organico"
            else _codigo_de_centro(vector_phasor, centros_fusion)
        )

    canales: dict[str, np.ndarray] = {"intensidad": intensidad}
    con_senal = peso > 0.08
    for canal, acum in acumulador.items():
        arr = np.full(forma, np.nan)
        arr[con_senal] = acum[con_senal] / peso[con_senal]
        canales[canal] = arr

    pila = np.stack(contribuciones)
    idx_blob = pila.argmax(axis=0)
    es_particula = pila.max(axis=0) > 0.25
    labels_crudos = np.where(es_particula, idx_blob + 1, 0).astype(np.int32)

    # Descarta "medialunas" residuales: en un par en contacto, argmax deja al blob más
    # tenue una porción diminuta que no es un objeto detectable por separado.
    area_min_verdad = 40
    for etiqueta in np.unique(labels_crudos[labels_crudos > 0]):
        if (labels_crudos == etiqueta).sum() < area_min_verdad:
            labels_crudos[labels_crudos == etiqueta] = 0

    labels, adelante, _ = relabel_sequential(labels_crudos)
    polimero = {
        int(adelante[k + 1]): codigos[k]
        for k in range(len(codigos))
        if adelante[k + 1] > 0
    }
    return canales, {"labels": np.asarray(labels), "polimero": polimero}


def _codigo_de_centro(vector: np.ndarray, centros: dict[str, np.ndarray]) -> str:
    """Devuelve el polímero cuyo centroide de fusión coincide con ``vector``."""
    for codigo, centro in centros.items():
        if np.allclose(centro, vector):
            return codigo
    return "no_clasificable"


def generar_mascara_celular(
    forma: tuple[int, int],
    verdad: dict,
    *,
    fraccion_fagocitada: float = 0.6,
    radio_celula_px: float = 16.0,
    semilla: int = 0,
) -> np.ndarray:
    """Máscara sintética de células que fagocitaron parte de las partículas.

    Dibuja un disco (la célula) alrededor de una fracción de las partículas de polímero de
    la verdad de terreno. Sirve para probar
    :func:`napari_mp_classifier.segmentacion.restringir_a_mascara` en el flujo de fagocitos
    (monocitos / neutrófilos, Park et al. 2020): las partículas dentro de una célula son
    NR-MP fagocitado; las de afuera se descartan.

    Parameters
    ----------
    forma : tuple of int
        Forma de la imagen (igual que la de :func:`generar_imagen_muestra`).
    verdad : dict
        El ``verdad`` que devuelve :func:`generar_imagen_muestra`.
    fraccion_fagocitada : float, optional
        Fracción de partículas de polímero que quedan dentro de una célula. Por defecto ``0.6``.
    radio_celula_px : float, optional
        Radio del disco celular. Por defecto ``16``.
    semilla : int, optional

    Returns
    -------
    numpy.ndarray of bool
        Máscara de las células.
    """
    rng = np.random.default_rng(semilla)
    alto, ancho = forma
    yy, xx = np.mgrid[0:alto, 0:ancho].astype(float)
    mascara = np.zeros(forma, dtype=bool)

    labels = verdad["labels"]
    polimeros = [lbl for lbl, cod in verdad["polimero"].items() if cod != "no_clasificable"]
    rng.shuffle(polimeros)
    n_dentro = round(fraccion_fagocitada * len(polimeros))

    for etiqueta in polimeros[:n_dentro]:
        fila, col = np.argwhere(labels == etiqueta).mean(axis=0)
        radio = radio_celula_px * rng.uniform(0.9, 1.3)
        mascara |= (yy - fila) ** 2 + (xx - col) ** 2 <= radio**2
    return mascara
