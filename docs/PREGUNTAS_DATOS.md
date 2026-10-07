# Preguntas sobre los datos reales

Estas respuestas definen el esquema de calibración y la lógica de fusión FLIM+espectral.
Mientras tanto se avanza con datos sintéticos (Fase 1).

## Calibración (los 6 polímeros de referencia)

1. **Formato de entrega**: ¿imágenes crudas (`.sdt` para FLIM, `.czi` para espectral) o
   coordenadas de phasor ya calculadas (CSV / OME-TIF exportado de `napari-phasors`)?
2. **Réplicas por polímero**: ¿un archivo por polímero, o varios campos/adquisiciones?
   (afecta cómo se estima la dispersión de cada cluster).
   **Nota (auditoría 2026-10):** el umbral de rechazo (Hotelling) necesita al menos
   `p + 1` mediciones por polímero (3 en una modalidad, 5 en fusión). Con tan pocas el
   cluster queda mal estimado y la región de aceptación es enorme. Cuantas más réplicas,
   mejor; el informe marca la calibración como «con observaciones» por debajo de 15.
3. **Etiquetado**: ¿cómo viene identificado el polímero de cada archivo? (nombre de
   archivo, planilla, carpeta).

## Parámetros FLIM (`.sdt`)

4. **Frecuencia de modulación / repetición del láser** (MHz) — necesaria para convertir a
   lifetime y para `phasor_calibrate`.
   **PARCIAL (prueba LAMAE):** el encabezado del `.sdt` trae `frequency = 59,96 MHz`.
5. **Referencia de calibración**: fluoróforo y lifetime conocido (ns) de la imagen de
   referencia (p. ej. fluoresceína 4,0 ns, rodamina B 1,68 ns).
   **PENDIENTE — bloquea la calibración FLIM.** Los `.sdt` de ejemplo no incluyen una
   imagen de referencia; el phasor FLIM sale sin calibrar.
   **Alternativa implementada (2026-10-06):** calibración con la IRF como SPCImage
   (`calibrar_irf=True`, ver `docs/CALIBRACION_FLIM.md`). Con IRF sintética el error en τφ
   es de +2 a +6 % entre 1 y 4 ns. **Más simple que un fluoróforo:** medir la IRF una vez
   por sesión (reflexión del láser en el cubreobjetos) y pasarla con `irf=`.
6. **Nº de bins temporales** del histograma TCSPC y **armónico(s)** de interés.
   **PARCIAL (prueba LAMAE):** 1024 bins sobre un período. Armónico de interés a definir.

## Parámetros espectrales (`.czi`)

7. **Rango de longitudes de onda** del λ-stack (nm inicial/final) y **nº de canales**.
   **RESPONDIDO (prueba LAMAE):** 451–721 nm, 28 canales, paso 10 nm.
8. ¿El `.czi` es lambda-stack real (muchos canales equiespaciados) o pocas bandas?
   `phasor_from_signal` necesita ≥ 3 muestras equiespaciadas.
   **RESPONDIDO (prueba LAMAE):** λ-stack real, equiespaciado. El phasor espectral se
   calcula sin problemas y no necesita calibración.

## Correspondencia entre modalidades

9. ¿Los `.sdt` y `.czi` de una misma muestra están **registrados espacialmente**
   (mismo campo, misma grilla de píxeles) o son adquisiciones independientes?
   - Registrados → fusión **por píxel/ROI** (vector de 4 features: g_FLIM, s_FLIM, g_esp, s_esp).
   - Independientes → fusión **por cluster** (se comparan distribuciones, no partículas).
   **PARCIAL (prueba LAMAE):** los ejemplos vienen a distinta resolución (512² FLIM vs
   440² espectral) y sin registro → hoy solo permitiría fusión por cluster. Confirmar si
   se puede adquirir ambas del mismo campo con la misma grilla.

## Muestras a clasificar

10. ¿Hay imágenes con **verdad de terreno** (partículas de polímero conocido en matriz
    ambiental o con monocitos/neutrófilos) para calcular las métricas de clasificación?
    **Además, controles negativos** (materia orgánica / células sin MP, teñidas igual): son
    los que permiten fijar `confianza`. En el escenario realista de la auditoría, la
    autofluorescencia cercana a un polímero pasa el umbral en ~50 % de los casos.
11. Matrices previstas: ¿solo ambientales, solo cultivos celulares, ambas? ¿en qué orden
    de prioridad?

## Detectores FLIM (surgidas al calibrar LAMAE, 2026-10-06)

13. El `.sdt` trae **dos detectores** (dos módulos TCSPC, mismo campo): 3,1 M y 63,3 M
    fotones, con lifetimes distintos (τφ 2,88 vs. 2,69 ns). **¿Qué filtro de emisión tiene
    cada uno?** Si miden bandas distintas, son dos features FLIM y no uno. Hoy se usa el de
    más fotones.
14. ¿Se puede **medir la IRF en cada sesión**? Ver pregunta 5.

## Envejecimiento (Meyers et al. 2024)

12. ¿La calibración se hace **solo con polímero virgen** (limitación declarada) o el
    equipo tiene material **degradado artificialmente** para sumar al set de calibración?

    **RESPONDIDO (2026-09-02).** No se usa polímero virgen. La calibración se hace sobre
    polímero **envejecido de forma controlada**: abrasión mecánica + H₂O₂ (oxidativo /
    térmico), opcionalmente UV 1 h (fotoenvejecimiento), antes de teñir con Nile Red. Esto
    alinea calibración y muestra ambiental en el mismo estado de degradación y sortea el
    modo de falla de Meyers 2024. Análisis del riesgo residual (variabilidad del grado de
    envejecimiento) y recomendaciones en `docs/DECISION_CALIBRACION.md` y
    `docs/RESULTADOS_FASE5.md`.
    Pendiente: nº de lotes del estándar por polímero, y si se sumarán mediciones con
    distintos grados de envejecimiento (con/sin UV, distintos tiempos de H₂O₂).
