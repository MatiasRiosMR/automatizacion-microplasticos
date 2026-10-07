"""Tests de :mod:`napari_mp_classifier.cli`."""

import numpy as np
import pytest
from datos_sinteticos import generar_calibracion, generar_imagen_muestra

from napari_mp_classifier.cli import main


@pytest.fixture
def muestra_y_calibracion(tmp_path):
    canales, verdad = generar_imagen_muestra(semilla=6)
    ruta_npz = tmp_path / "muestra.npz"
    np.savez(ruta_npz, **canales)

    df_cal = generar_calibracion("fusion", n_por_polimero=60, semilla=0)
    ruta_cal = tmp_path / "cal.csv"
    df_cal.to_csv(ruta_cal, index=False)
    return ruta_npz, ruta_cal, verdad


def test_sin_comando_muestra_ayuda(capsys):
    assert main([]) == 0
    assert "classify" in capsys.readouterr().out


def test_classify_genera_reporte(muestra_y_calibracion, tmp_path):
    ruta_npz, ruta_cal, _ = muestra_y_calibracion
    salida = tmp_path / "out"
    codigo = main([
        "classify", str(ruta_npz),
        "--calibracion", str(ruta_cal),
        "--salida", str(salida),
        "--escala-um-px", "0.18",
    ])
    assert codigo == 0
    assert (salida / "asignaciones.csv").exists()
    assert (salida / "resumen_muestra.md").exists()
    assert (salida / "figuras" / "phasores_muestra.png").exists()
    assert (salida / "figuras" / "segmentacion.png").exists()


def test_classify_estrategia_centroide(muestra_y_calibracion, tmp_path):
    ruta_npz, ruta_cal, _ = muestra_y_calibracion
    codigo = main([
        "classify", str(ruta_npz),
        "--calibracion", str(ruta_cal),
        "--salida", str(tmp_path / "out"),
        "--estrategia", "centroide",
        "--modalidad", "fusion",
    ])
    assert codigo == 0


def test_classify_calibracion_sin_columnas(tmp_path, muestra_y_calibracion):
    ruta_npz, _, _ = muestra_y_calibracion
    mala = tmp_path / "mala.csv"
    mala.write_text("a,b\n1,2\n", encoding="utf-8")
    codigo = main([
        "classify", str(ruta_npz), "--calibracion", str(mala),
        "--salida", str(tmp_path / "out"),
    ])
    assert codigo == 2


def test_classify_muestra_inexistente(tmp_path, muestra_y_calibracion):
    _, ruta_cal, _ = muestra_y_calibracion
    codigo = main([
        "classify", str(tmp_path / "no_existe.npz"),
        "--calibracion", str(ruta_cal),
        "--salida", str(tmp_path / "out"),
    ])
    assert codigo == 2


def test_classify_modalidad_espectral_con_npz_de_cuatro_canales(muestra_y_calibracion, tmp_path):
    # Antes fallaba: el pipeline volvía a deducir 'fusion' de los canales.
    ruta_npz, _, _ = muestra_y_calibracion
    df = generar_calibracion("espectral", n_por_polimero=60, semilla=0)
    ruta_cal = tmp_path / "cal_esp.csv"
    df.to_csv(ruta_cal, index=False)
    salida = tmp_path / "esp"
    codigo = main(["classify", str(ruta_npz), "--calibracion", str(ruta_cal),
                   "--salida", str(salida), "--modalidad", "espectral"])
    assert codigo == 0
    assert (salida / "asignaciones.csv").exists()


def test_classify_genera_informe_html(muestra_y_calibracion, tmp_path):
    ruta_npz, ruta_cal, _ = muestra_y_calibracion
    salida = tmp_path / "inf"
    assert main(["classify", str(ruta_npz), "--calibracion", str(ruta_cal),
                 "--salida", str(salida), "--nombre", "Muestra X", "--sin-pdf"]) == 0
    html = (salida / "informe.html").read_text(encoding="utf-8")
    assert "Muestra X" in html and "Informe de análisis de microplásticos" in html


def test_classify_sin_informe(muestra_y_calibracion, tmp_path):
    ruta_npz, ruta_cal, _ = muestra_y_calibracion
    salida = tmp_path / "sin"
    assert main(["classify", str(ruta_npz), "--calibracion", str(ruta_cal),
                 "--salida", str(salida), "--sin-informe"]) == 0
    assert not (salida / "informe.html").exists()


def test_classify_con_calibracion_json(muestra_y_calibracion, tmp_path):
    from datos_sinteticos import _columnas

    from napari_mp_classifier import Calibracion

    ruta_npz, _, _ = muestra_y_calibracion
    df = generar_calibracion("fusion", n_por_polimero=60, semilla=0)
    ruta_json = tmp_path / "cal.json"
    Calibracion.desde_dataframe(df, columnas=_columnas("fusion")).guardar_json(ruta_json)
    salida = tmp_path / "json"
    args = ["classify", str(ruta_npz), "--calibracion", str(ruta_json), "--salida", str(salida),
            "--sin-informe"]
    assert main([*args, "--estrategia", "centroide"]) == 0
    assert main([*args, "--estrategia", "knn"]) == 2  # knn necesita las mediciones (CSV)
