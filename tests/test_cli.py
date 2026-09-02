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
