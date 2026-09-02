"""Tests de :mod:`napari_mp_classifier.fusion`."""

import numpy as np
import pandas as pd
import pytest

from napari_mp_classifier import NO_CLASIFICABLE
from napari_mp_classifier.fusion import fusionar_por_decision, fusionar_por_roi


def _features(sufijo, centros, ruido=0.0, semilla=0):
    """DataFrame de features mínimo (phasor + centroide) para n ROIs."""
    rng = np.random.default_rng(semilla)
    filas = []
    for i, (fila, col, g, s) in enumerate(centros, start=1):
        filas.append({
            "label": i,
            "centro_fila": fila + rng.normal(0, ruido),
            "centro_col": col + rng.normal(0, ruido),
            f"g_{sufijo}": g,
            f"s_{sufijo}": s,
            f"dispersion_{sufijo}": 0.01,
            "area_px": 100.0,
        })
    return pd.DataFrame(filas).set_index("label")


def test_fusionar_por_roi_empareja_por_centroide():
    flim = _features("flim", [(10, 10, 0.3, 0.4), (50, 50, 0.6, 0.3)])
    esp = _features("esp", [(51, 49, 0.5, 0.5), (11, 9, 0.2, 0.35)], semilla=1)

    fusion = fusionar_por_roi(flim, esp, tol_centro_px=5.0)
    assert len(fusion) == 2
    assert list(fusion.columns[:3]) == ["label_flim", "label_esp", "dist_centro_px"]
    # La ROI FLIM en (10,10) se empareja con la ROI esp en (11,9).
    fila = fusion[fusion["label_flim"] == 1].iloc[0]
    assert fila["label_esp"] == 2
    assert fila["g_flim"] == 0.3 and fila["g_esp"] == 0.2
    assert {"g_flim", "s_flim", "g_esp", "s_esp"}.issubset(fusion.columns)


def test_fusionar_por_roi_descarta_sin_pareja():
    flim = _features("flim", [(10, 10, 0.3, 0.4), (200, 200, 0.6, 0.3)])
    esp = _features("esp", [(10, 10, 0.5, 0.5)])
    fusion = fusionar_por_roi(flim, esp, tol_centro_px=5.0)
    assert len(fusion) == 1
    assert fusion.iloc[0]["label_flim"] == 1


def test_fusionar_por_roi_columnas_faltantes():
    flim = _features("flim", [(10, 10, 0.3, 0.4)])
    esp = pd.DataFrame({"g_esp": [0.5], "s_esp": [0.5]})
    with pytest.raises(ValueError, match="centro"):
        fusionar_por_roi(flim, esp)


def test_fusionar_por_decision_acuerdo_y_desacuerdo():
    pred_flim = np.array(["PET", "HDPE", "PVC", "PP"])
    pred_esp = np.array(["PET", "LDPE", "PVC", NO_CLASIFICABLE])
    combinada = fusionar_por_decision(pred_flim, pred_esp)
    assert list(combinada) == ["PET", NO_CLASIFICABLE, "PVC", NO_CLASIFICABLE]


def test_fusionar_por_decision_score_alto_rechaza():
    pred_flim = np.array(["PET", "PET"])
    pred_esp = np.array(["PET", "PET"])
    combinada = fusionar_por_decision(
        pred_flim, pred_esp, score_flim=np.array([0.5, 2.0])
    )
    assert list(combinada) == ["PET", NO_CLASIFICABLE]


def test_fusionar_por_decision_longitudes_distintas():
    with pytest.raises(ValueError, match="longitud"):
        fusionar_por_decision(["PET"], ["PET", "PS"])
