# INFORMES.md — informe por muestra (HTML + PDF)

`informe_html.generar_informe_html` convierte el resultado de `analizar_muestra` en un
informe con formato de **informe de ensayo de laboratorio**, pensado para el investigador
que no programa. La CLI lo genera solo por cada muestra.

## Cómo se genera

Desde la CLI (por defecto):

```bash
napari-mp-classifier classify muestra.npz --calibracion calibracion.json \
    --salida resultados/ --nombre M-01 --escala-um-px 0.0767
```

Escribe `resultados/informe.html` y, si hay Chrome/Chromium instalado,
`resultados/informe.pdf`. `--sin-pdf` omite el PDF y `--sin-informe` omite ambos.

Desde Python:

```python
from napari_mp_classifier import analizar_muestra
from napari_mp_classifier.informe_html import generar_informe_html

res = analizar_muestra(canales, calibracion, escala_um_px=0.0767)
generar_informe_html(res, canales, "informe_M-01.html", nombre_muestra="M-01",
                     archivo="M-01.npz")
```

Ejemplos:

- `python ejemplos/demo_informe.py` → `ejemplos/salida_demo_informes/`: **M-01** (control de
  composición conocida, con sección de verificación) y **M-02** (muestra «ambiental» sin
  verdad de terreno, algo más envejecida que la calibración).
- `python ejemplos/informe_lamae.py` → `ejemplos/salida_demo_lamae/informe_LAMAE_PE.html`:
  informe de **caracterización** (sin clasificación) de la partícula real de LAMAE, con
  espectro, decaimiento, IRF estimada y FLIM calibrado en los dos detectores.

## Estructura

1. **Portada**: membrete (UNER · CONICET — LAMAE · LaSBI), número de informe (fecha + huella
   SHA-256 del archivo), estado del análisis, 5 indicadores clave y barra de composición.
2. **Muestra y condiciones**: archivo, modalidad, clasificador, calibración, escala.
3. **Resultados**:
   - composición por polímero (n, fracción con **IC 95 % de Wilson**, área, diámetro
     mediano);
   - imagen con las partículas numeradas y barra de escala;
   - diagramas de phasores FLIM y espectral lado a lado, con la región de aceptación de
     cada polímero;
   - tamaño por polímero y score de rechazo;
   - partículas con observaciones y tabla por partícula.
4. **Verificación**: solo si hay verdad de terreno (`polimero_real`). Incluye indicadores y
   matriz de confusión.
5. **Observaciones, método, alcance y limitaciones, firmas** (Analizó / Revisó / Aprobó).
6. **Anexos**: A. calibración de referencia y trazabilidad (SHA-256, versiones); B. glosario
   con solo los términos que aparecen en el informe (`glosario.py`).

## Estado del análisis

Se calcula solo:

- **No concluyente**: no se detectaron partículas.
- **Válido con observaciones**: se cumple alguna de estas condiciones:
  - la calibración tiene menos de 15 mediciones por polímero;
  - la calibración no registra metadatos (fecha, equipo, protocolo de envejecimiento);
  - más del 30 % de las partículas resultó no clasificable;
  - hay partículas para revisar a mano;
  - la escala no se informó o no está medida.
- **Válido**: ninguna de las anteriores.

Una partícula queda **para revisar a mano** si su phasor es inválido (NaN), si tiene un
score entre 0,7 y 1 (cerca del umbral) o si su phasor es heterogéneo (dispersión intra-ROI
por encima del percentil 90 o de 0,05: posible mezcla o dos partículas juntas). Hay además
observaciones informativas: toca el borde, forma irregular o tamaño muy pequeño.

## En pantalla y en papel

- **En pantalla**: la imagen de partículas, los diagramas y la tabla están vinculados, así
  que un clic en cualquiera marca la misma partícula en los otros. La tabla tiene buscador,
  filtro por polímero y «Solo con observaciones». Cada término técnico tiene su ayuda «?» y
  los datos se descargan en CSV. Los diagramas interactivos usan Plotly desde CDN: sin
  internet ese recuadro queda vacío y todo lo demás funciona.
- **PDF**: `exportar_a_pdf` imprime con Chrome headless (`--no-pdf-header-footer`) en A4,
  con pie propio «Informe N.º · Página X de Y». Oculta los controles y las ayudas y
  reemplaza los diagramas interactivos por figuras estáticas. Si no hay Chrome devuelve
  `None` y solo queda el HTML. La variable `NAPARI_MP_SIN_PDF` lo desactiva (la usan los
  tests).

## Limitaciones

- Con calibraciones sintéticas el informe hereda sus supuestos: los valores son
  ilustrativos.
- La escala de los ejemplos sintéticos es nominal y el informe lo aclara (`nota_escala`).
- El informe de LAMAE no clasifica: falta la calibración de los 6 polímeros.
