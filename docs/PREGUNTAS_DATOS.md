# Preguntas sobre los datos reales

Estas respuestas definen el esquema de calibración y la lógica de fusión FLIM+espectral.
Mientras tanto se avanza con datos sintéticos (Fase 1).

## Calibración (los 6 polímeros de referencia)

1. **Formato de entrega**: ¿imágenes crudas (`.sdt` para FLIM, `.czi` para espectral) o
   coordenadas de phasor ya calculadas (CSV / OME-TIF exportado de `napari-phasors`)?
2. **Réplicas por polímero**: ¿un archivo por polímero, o varios campos/adquisiciones?
   (afecta cómo se estima la dispersión de cada cluster).
3. **Etiquetado**: ¿cómo viene identificado el polímero de cada archivo? (nombre de
   archivo, planilla, carpeta).

## Parámetros FLIM (`.sdt`)

4. **Frecuencia de modulación / repetición del láser** (MHz) — necesaria para convertir a
   lifetime y para `phasor_calibrate`.
5. **Referencia de calibración**: fluoróforo y lifetime conocido (ns) de la imagen de
   referencia (p. ej. fluoresceína 4,0 ns, rodamina B 1,68 ns).
6. **Nº de bins temporales** del histograma TCSPC y **armónico(s)** de interés.

## Parámetros espectrales (`.czi`)

7. **Rango de longitudes de onda** del λ-stack (nm inicial/final) y **nº de canales**.
8. ¿El `.czi` es lambda-stack real (muchos canales equiespaciados) o pocas bandas?
   `phasor_from_signal` necesita ≥ 3 muestras equiespaciadas.

## Correspondencia entre modalidades

9. ¿Los `.sdt` y `.czi` de una misma muestra están **registrados espacialmente**
   (mismo campo, misma grilla de píxeles) o son adquisiciones independientes?
   - Registrados → fusión **por píxel/ROI** (vector de 4 features: g_FLIM, s_FLIM, g_esp, s_esp).
   - Independientes → fusión **por cluster** (se comparan distribuciones, no partículas).

## Muestras a clasificar

10. ¿Hay imágenes con **verdad de terreno** (partículas de polímero conocido en matriz
    ambiental o con monocitos/neutrófilos) para calcular las métricas de clasificación?
11. Matrices previstas: ¿solo ambientales, solo cultivos celulares, ambas? ¿en qué orden
    de prioridad?

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
