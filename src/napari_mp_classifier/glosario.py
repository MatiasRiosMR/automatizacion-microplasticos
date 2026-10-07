"""Glosario y ayudas contextuales «?» de los informes HTML.

Cada término técnico que aparece en un informe (phasor, IRF, score de rechazo, F1, …) tiene
una definición en lenguaje llano en :data:`GLOSARIO`. :func:`ayuda` devuelve el botón «?»
accesible que la muestra al pasar el cursor, al enfocarlo con el teclado o al tocarlo
(móvil); :data:`CSS_AYUDA` y :data:`JS_AYUDA` se insertan una vez por página. Los mismos
textos forman el anexo «Glosario» de cada informe.
"""

from __future__ import annotations

import html

GLOSARIO: dict[str, tuple[str, str]] = {
    # ---------------------------------------------------------------- phasores y modalidades
    "phasor": ("Phasor",
               ("Una forma de resumir la señal de cada píxel (el decaimiento FLIM o el espectro) en "
               "un solo punto (g, s) de un plano. No hace falta ajustar curvas: materiales con una "
               "emisión parecida caen cerca entre sí. Clasificar es ver en qué zona del plano cae cada partícula.")),
    "gs": ("Coordenadas g y s",
           ("Las dos coordenadas del phasor: g es la parte «coseno» y s la parte «seno» del primer "
           "armónico de la señal normalizada. Por sí solas no tienen unidades; lo que importa es la "
           "posición relativa entre partículas y referencias.")),
    "flim": ("FLIM",
             ("Imagen de tiempo de vida de fluorescencia: mide cuánto tarda la fluorescencia en apagarse "
             "después de cada pulso láser (nanosegundos). El Nile Red cambia su tiempo de vida según la "
             "polaridad y rigidez del polímero que lo rodea; por eso sirve para distinguir plásticos.")),
    "espectral": ("Phasor espectral (λ-stack)",
                  ("La imagen se adquiere en muchas bandas de color (aquí 28 canales entre 451 y 721 nm). "
                  "El phasor espectral resume dónde está el máximo de emisión (ángulo) y qué tan ancho es "
                  "el espectro (distancia al centro). No necesita calibración: la longitud de onda es absoluta.")),
    "fusion": ("Fusión FLIM + espectral (4D)",
               ("Se usan a la vez las coordenadas FLIM (g, s) y las espectrales (g, s): cada partícula es un "
               "punto en 4 dimensiones. Dos polímeros que se superponen en un plano pueden separarse en el otro.")),
    "semicirculo": ("Semicírculo universal",
                    ("En FLIM, cualquier decaimiento de un solo tiempo de vida cae exactamente sobre este "
                    "semicírculo; las marcas indican el lifetime en ns (a la derecha, cortos; a la izquierda, "
                    "largos). Si la señal mezcla varios tiempos de vida, el punto cae adentro.")),
    "circulo_espectral": ("Círculo espectral",
                          ("En el phasor espectral el ángulo indica la longitud de onda del centro del espectro "
                          "(marcas en nm) y la distancia al origen, qué tan angosto es el espectro.")),
    "tau": ("τφ y τm (lifetimes aparentes)",
            ("Tiempos de vida calculados desde la fase (τφ) y la modulación (τm) del phasor. Si coinciden, la "
            "emisión es de un solo exponencial; si τφ < τm hay una mezcla de tiempos de vida (punto "
            "dentro del semicírculo).")),
    "irf": ("IRF (respuesta instrumental)",
            ("La forma temporal del pulso láser + detector + electrónica. Corre y ensancha el decaimiento "
            "medido: sin descontarla, el phasor queda rotado y los lifetimes salen mal. Puede medirse "
            "(reflexión en el cubreobjetos, SHG de un cristal) o estimarse del propio decaimiento.")),
    "irf_flanco": ("IRF sintética del flanco (como SPCImage)",
                   ("SPCImage estima la IRF a partir del flanco de subida del decaimiento: ahí la derivada de la "
                   "curva es casi igual a la IRF. Después corrige el phasor dividiéndolo por el phasor de esa IRF "
                   "(se le resta su fase y se divide por su modulación). Acá se hace igual, con phasorpy.")),
    "pico0": ("Corrección «pico → 0» (método actual del repo)",
              ("Rota el decaimiento para que el máximo quede en el bin 0. Acerca la nube al semicírculo pero "
              "ignora que el máximo llega después del pulso: subestima los lifetimes ~15–20 %.")),
    "tac": ("Ventana del TAC",
            ("El convertidor tiempo-amplitud solo registra fotones entre sus límites inferior y superior (acá "
            "5 % y 96 % del rango). Los bins fuera de la ventana quedan en cero y se pierde un poco de la cola "
            "del decaimiento, lo que sesga lifetimes largos (> 5 ns).")),
    "detectores": ("Dos detectores en el .sdt",
                   ("El archivo tiene dos bloques de datos de dos placas TCSPC (series 3N0140 y 3N0141): el mismo "
                   "campo visto por dos detectores al mismo tiempo, probablemente dos bandas de emisión. Hay que "
                   "confirmar con el equipo qué mide cada uno.")),
    "fotones": ("Fotones por píxel",
                ("Cuántos fotones se contaron en cada píxel. Con menos de ~100 el phasor de un píxel es muy "
                "ruidoso; por eso se filtra (mediana) o se agrupa por partícula.")),
    "mediana_filtro": ("Filtro de mediana del phasor",
                       ("Reemplaza g y s de cada píxel por la mediana de sus vecinos 3×3 (dos pasadas). Reduce el "
                       "ruido de pocos fotones sin borrar bordes. Es el filtro estándar de phasorpy y SPCImage.")),
    # ---------------------------------------------------------------- clasificación
    "calibracion_ref": ("Calibración de referencia",
                        ("Mediciones de partículas de cada uno de los 6 polímeros conocidos (envejecidas con el mismo "
                        "protocolo que las muestras). Definen dónde cae cada polímero en el plano de phasores. Sin "
                        "esto no hay contra qué clasificar.")),
    "n_cal": ("n mediciones",
              ("Cuántas partículas de referencia respaldan a cada polímero. Con pocas (menos de ~15) la zona de "
              "cada polímero está mal estimada y el umbral debe ensancharse para no rechazar de más.")),
    "elipse": ("Elipses de referencia",
               ("Relleno: dispersión típica de cada polímero en la calibración (1 desvío). Trazo punteado: región "
               "de aceptación; una partícula que cae fuera de todas se marca como no clasificable.")),
    "score": ("Score de rechazo",
              ("Distancia de la partícula a su polímero asignado dividida por el umbral. 0 = justo en el centro; "
              "1 = en el borde de la región de aceptación; más de 1 = no clasificable. Entre 0,7 y 1 la "
              "asignación es menos segura.")),
    "confianza": ("Confianza",
                  ("Fracción de partículas de un polímero de referencia que se espera aceptar. Con 0,99, en teoría, "
                  "solo 1 de cada 100 partículas reales queda como no clasificable. Subirla acepta más (y deja "
                  "pasar más materia orgánica); bajarla rechaza más.")),
    "no_clasif": ("No clasificable",
                  ("La firma de phasor cae fuera de las zonas de los 6 polímeros de referencia: materia orgánica, "
                  "autofluorescencia de células, una mezcla, o un plástico que no está en el panel. No significa "
                  "«no es plástico».")),
    "umbral_chi2": ("Umbral χ² (actual)",
                    ("Umbral que supone que el centro y la forma de cada polímero se conocen exactamente. Con pocas "
                    "mediciones de referencia eso no es cierto y rechaza muchas más partículas reales de lo que "
                    "promete la «confianza».")),
    "hotelling": ("Umbral de Hotelling (propuesto)",
                  ("Igual que el χ² pero tiene en cuenta que el centro y la forma de cada polímero se estimaron con n "
                  "mediciones: el umbral se ensancha cuando n es chico y converge al χ² cuando n es grande.")),
    "qda": ("QDA",
            ("Análisis discriminante cuadrático: asigna cada partícula al polímero con mayor verosimilitud "
            "gaussiana, teniendo en cuenta la forma y el tamaño de cada nube (no solo la distancia al centro).")),
    "centroide": ("Estrategia «centroide»",
                  ("Asigna al polímero cuyo centro está más cerca, midiendo la distancia en unidades de la "
                  "dispersión de cada polímero (distancia de Mahalanobis).")),
    "knn": ("Estrategia «knn»",
            ("Mira las k mediciones de referencia más cercanas a la partícula y vota. No supone nubes "
            "elípticas, pero su umbral de rechazo es más difícil de calibrar.")),
    "gmm": ("Estrategia «gmm»",
            ("Modela cada polímero como una nube gaussiana (mezcla de gaussianas). En el repo se ajusta sin usar "
            "las etiquetas, lo que puede mezclar polímeros.")),
    "envejecimiento": ("Envejecimiento",
                       ("Los plásticos ambientales están degradados (abrasión, oxidación, UV) y su firma con Nile Red "
                       "se corre. Si la muestra está más (o menos) envejecida que la calibración, sus partículas se "
                       "alejan de las referencias. 0 = mismo estado que la calibración.")),
    # ---------------------------------------------------------------- métricas
    "exactitud": ("Exactitud",
                  "Fracción de partículas asignadas correctamente (incluye acertar «no clasificable»)."),
    "exactitud_bal": ("Exactitud balanceada",
                      ("Promedio del acierto de cada clase. A diferencia de la exactitud, no se infla cuando una "
                      "clase tiene muchas más partículas que las otras.")),
    "precision": ("Precisión",
                  ("De las partículas asignadas a un polímero, cuántas lo eran de verdad. Baja precisión = "
                  "falsos positivos.")),
    "recall": ("Recall (sensibilidad)",
               ("De las partículas que realmente eran de un polímero, cuántas se encontraron. Bajo recall = "
               "partículas perdidas o mal asignadas.")),
    "f1": ("F1",
           ("Media armónica entre precisión y recall: resume ambas en un número. «Macro» = promedio entre "
           "clases, cada una con el mismo peso.")),
    "matriz": ("Matriz de confusión",
               ("Filas = clase real, columnas = clase asignada. La diagonal son los aciertos; todo lo que está "
               "fuera de la diagonal es un error y dice con qué se confundió.")),
    "roc": ("Curva ROC y AUROC",
            ("Muestra el compromiso entre rechazar materia orgánica (eje vertical) y rechazar por error polímero "
            "real (eje horizontal) al mover el umbral. AUROC = área bajo la curva: 1 = separación perfecta, "
            "0,5 = azar. No depende del umbral elegido.")),
    "falso_rechazo": ("Falso rechazo",
                      "Partículas de polímero real que terminaron como «no clasificable»."),
    "organico_aceptado": ("Orgánico aceptado",
                          ("Materia orgánica o autofluorescencia que el clasificador asignó a un polímero: es un "
                          "falso positivo de microplástico.")),
    "ic": ("Intervalo de confianza 95 % (bootstrap)",
           ("Rango en el que probablemente está el valor real de la métrica. Se calcula remuestreando las "
           "partículas 600 veces. Si es ancho, hay pocas partículas para afirmar mucho.")),
    "ic_wilson": ("Intervalo de confianza 95 % de una fracción (Wilson)",
                  ("Rango en el que probablemente está la fracción real de cada polímero en la muestra, dado el "
                  "número de partículas analizadas. Con pocas partículas el intervalo es ancho: la fracción "
                  "observada es solo una estimación.")),
    "sd": ("± sd",
           ("Desvío estándar entre repeticiones con distintas semillas aleatorias: cuánto varía el resultado "
           "de una muestra a otra.")),
    "iou": ("IoU",
            ("Intersección sobre unión entre el contorno detectado y el real: 1 = idénticos, 0 = no se tocan. "
            "Mide qué tan bien se dibujó cada partícula.")),
    "f1_det": ("F1 de detección",
               "Combina cuántas partículas reales se encontraron y cuántas detecciones eran partículas reales."),
    # ---------------------------------------------------------------- imagen y partículas
    "segmentacion": ("Segmentación",
                     ("Paso que encuentra las partículas en la imagen y dibuja su contorno (ROI). Acá: umbral de "
                     "Otsu sobre la intensidad y watershed para separar partículas que se tocan.")),
    "watershed": ("Watershed",
                  ("Algoritmo que separa partículas en contacto «inundando» desde el centro de cada una. Puede "
                  "partir una partícula alargada en dos (sobre-segmentar).")),
    "roi": ("ROI / partícula",
            ("Región de interés: el conjunto de píxeles que se considera una partícula. Su phasor es la mediana "
            "de los phasores de sus píxeles.")),
    "dispersion": ("Phasor heterogéneo (dispersión)",
                   ("Cuánto varía el phasor entre los píxeles de una misma partícula. Si es alto, la partícula "
                   "puede ser una mezcla de materiales o dos partículas pegadas.")),
    "diametro": ("Diámetro equivalente",
                 ("Diámetro del círculo con la misma área que la partícula. Sirve para comparar tamaños aunque las "
                 "partículas no sean redondas.")),
    "aspecto": ("Relación de aspecto",
                "Eje mayor / eje menor. 1 = redonda; valores altos = fibra o fragmento alargado."),
    "intensidad": ("Intensidad media",
                   "Brillo promedio de Nile Red dentro de la partícula (unidades del equipo)."),
    "revisar": ("Partículas para revisar",
                ("Partículas cuya asignación conviene mirar a mano: cerca del umbral, con phasor heterogéneo o con "
                "phasor inválido.")),
    "huella": ("Huella del archivo",
               ("Código calculado a partir del contenido del archivo analizado (SHA-256). Si el archivo cambia, la "
               "huella cambia: permite saber exactamente qué datos produjeron este informe.")),
    "area_total": ("Área total de microplástico",
                   "Suma del área de todas las partículas asignadas a un polímero (excluye no clasificables)."),
    "escala": ("Escala",
               "Tamaño físico de un píxel. Sin escala los tamaños quedan en píxeles."),
}

CSS_AYUDA = """
.ayuda { display:inline-flex; align-items:center; justify-content:center; width:17px; height:17px; margin-left:5px;
  border-radius:50%; border:1px solid #b9b8b0; background:#fff; color:#52514e; font:700 11px/1 Lato,system-ui,sans-serif;
  cursor:help; vertical-align:middle; padding:0; text-transform:none; letter-spacing:0; flex:none; }
.ayuda:hover, .ayuda:focus-visible, .ayuda[aria-expanded="true"] { background:#2a78d6; border-color:#2a78d6; color:#fff; outline:none; }
#globo-ayuda { position:absolute; z-index:50; max-width:340px; background:#1f1f1d; color:#f4f3ee; border-radius:8px;
  padding:10px 13px; font:400 13.5px/1.5 Lato,system-ui,sans-serif; box-shadow:0 6px 24px rgba(0,0,0,.18);
  display:none; text-transform:none; letter-spacing:0; }
#globo-ayuda b { display:block; color:#fff; margin-bottom:3px; font-size:13.5px; }
@media print { .ayuda, #globo-ayuda { display:none !important; } }
"""

JS_AYUDA = """
(() => {
  const globo = document.createElement('div'); globo.id = 'globo-ayuda'; globo.setAttribute('role', 'tooltip');
  document.body.appendChild(globo);
  let fijo = null;
  function mostrar(b) {
    globo.textContent = ''; const t = document.createElement('b'); t.textContent = b.dataset.titulo;
    globo.append(t, document.createTextNode(b.dataset.texto));
    globo.style.display = 'block';
    const r = b.getBoundingClientRect(), w = globo.offsetWidth, h = globo.offsetHeight;
    let x = r.left + window.scrollX + r.width / 2 - w / 2;
    x = Math.max(8 + window.scrollX, Math.min(x, window.scrollX + document.documentElement.clientWidth - w - 8));
    let y = r.bottom + window.scrollY + 8;
    if (r.bottom + h + 16 > window.innerHeight) y = r.top + window.scrollY - h - 8;
    globo.style.left = x + 'px'; globo.style.top = y + 'px';
    b.setAttribute('aria-describedby', 'globo-ayuda');
  }
  function ocultar() { globo.style.display = 'none'; document.querySelectorAll('.ayuda[aria-expanded="true"]').forEach(x => x.setAttribute('aria-expanded', 'false')); }
  document.querySelectorAll('.ayuda').forEach(b => {
    b.addEventListener('mouseenter', () => { if (!fijo) mostrar(b); });
    b.addEventListener('mouseleave', () => { if (!fijo) ocultar(); });
    b.addEventListener('focus', () => mostrar(b));
    b.addEventListener('blur', () => { if (fijo !== b) ocultar(); });
    b.addEventListener('click', e => { e.stopPropagation(); e.preventDefault();
      if (fijo === b) { fijo = null; ocultar(); } else { fijo = b; mostrar(b); b.setAttribute('aria-expanded', 'true'); } });
  });
  document.addEventListener('click', () => { if (fijo) { fijo = null; ocultar(); } });
  document.addEventListener('keydown', e => { if (e.key === 'Escape') { fijo = null; ocultar(); } });
  window.addEventListener('scroll', () => { if (!fijo) ocultar(); }, { passive: true });
})();
"""


def ayuda(clave: str) -> str:
    """Botón «?» accesible con la definición de un término del glosario.

    Parameters
    ----------
    clave : str
        Clave de :data:`GLOSARIO` (p. ej. ``"phasor"``, ``"score"``, ``"irf_flanco"``).

    Returns
    -------
    str
        HTML del botón (``<button class="ayuda">``).

    Raises
    ------
    KeyError
        Si ``clave`` no está en el glosario.
    """
    titulo, texto = GLOSARIO[clave]
    return (f'<button type="button" class="ayuda" aria-label="Ayuda: {html.escape(titulo)}" '
            f'aria-expanded="false" data-titulo="{html.escape(titulo)}" '
            f'data-texto="{html.escape(texto)}">?</button>')
