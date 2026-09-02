# PIPELINE.md — flujo de datos de napari-mp-classifier

```mermaid
flowchart TD
    subgraph CAL["Calibración (una vez)"]
        A1[".sdt / .czi de los 6 polímeros<br/>(o CSV de phasores ya calculados)"]
        A2["phasorpy: signal_from_sdt / signal_from_czi<br/>+ phasor_from_signal + phasor_calibrate"]
        A3["Calibracion:<br/>centroide + covarianza por polímero"]
        A1 --> A2 --> A3
    end

    subgraph MUE["Análisis de una muestra"]
        B1["Imagen de muestra NR<br/>(ambiental / con células)"]
        B2["segmentacion.py<br/>Otsu / K-means + watershed + máscara celular"]
        B3["features.py<br/>phasor medio, intensidad, tamaño, forma por ROI"]
        B4["fusion.py<br/>[g_flim, s_flim, g_esp, s_esp] por ROI"]
        B1 --> B2 --> B3 --> B4
    end

    A3 --> C1
    B4 --> C1["clasificador.py<br/>centroide Mahalanobis / KNN / GMM"]
    C1 --> C2{"score > umbral?"}
    C2 -- sí --> C3["no_clasificable<br/>(materia orgánica / autofluorescencia)"]
    C2 -- no --> C4["PET / HDPE / PVC / LDPE / PP / PS"]

    C3 --> D1["reportes.py"]
    C4 --> D1
    D1 --> D2["CSV de asignaciones"]
    D1 --> D3["diagrama de phasores<br/>clusters ref + partículas"]
    D1 --> D4["overlay de imagen con ROIs etiquetadas"]
    D1 --> D5["métricas: exactitud, precisión,<br/>recall, F1, matriz de confusión"]
```

## Estado de implementación por etapa

| Etapa | Módulo | Fase | Estado |
|---|---|---|---|
| Lectura de crudo → phasores por píxel | `io_crudo.py` | 0+ | pendiente (necesita datos reales) |
| Spectral unmixing NR-MP vs autofluorescencia | `desmezcla.py` | 2-3 | pendiente — usa `phasorpy.component` (ver `docs/SPECTRAL_UNMIXING.md`) |
| Calibración (centroide + covarianza) | `calibracion.py` | 1 | **hecho** |
| Clasificación + "no clasificable" | `clasificador.py` | 1 | **hecho** (centroide / KNN / GMM) |
| Métricas estándar | `metricas.py` | 1 | **hecho** |
| Segmentación de ROIs | `segmentacion.py` | 2 | **hecho** (Otsu / K-means FIMAP + watershed + máscara celular; IoU 0,81) |
| Features por ROI | `features.py` | 2 | **hecho** (phasor por ROI + forma + intensidad + dispersión) |
| Métricas de segmentación | `metricas.py` | 2 | **hecho** (IoU, precisión/recall de detección) |
| Fusión FLIM + espectral | `fusion.py` | 3 | **hecho** (`fusionar_por_roi` registrado, `fusionar_por_decision` no registrado) |
| Desmezcla NR-MP vs autofluorescencia | `desmezcla.py` | 3 | **hecho** (envuelve `phasorpy.component`; componente autofluor. sintético hasta tener `.czi`) |
| Pipeline seg→features→clasificación | `pipeline.py` | 3 | **hecho** (`analizar_muestra` → `ResultadoMuestra`) |
| CSV + gráficos + resumen + informe unificado | `reportes.py` | 3 | **hecho** (`generar_reporte`: CSV + métricas + figuras + `resumen_muestra.md`) |
| CLI `classify` | `cli.py` | 3 | **hecho** (sobre `.npz`; `.sdt`/`.czi` esperan `io_crudo`) |
| Plugin napari | `napari_integracion/` | 4 | **hecho** (widget de clasificación + phasor plot con back-projection; entorno `napari-mp-env`, py 3.12) |

## Regla de "no clasificable" (detalle)

Ver docstring de `clasificador.py`. Resumen: una partícula queda `no_clasificable` cuando
su distancia de Mahalanobis al cuadrado al cluster de polímero más cercano supera
`chi2.ppf(confianza, df=n_features)` (con `confianza=0.99` por defecto). El umbral es
**consciente de la dimensión** (2 features para una modalidad, 4 para la fusión
FLIM+espectral), a diferencia de un umbral fijo en σ. Es la barrera contra falsos
positivos por materia orgánica fluorescente (muestras ambientales) y autofluorescencia
celular (monocitos / neutrófilos), y absorbe la deriva por envejecimiento
(Meyers et al. 2024) al bajar `confianza`.
