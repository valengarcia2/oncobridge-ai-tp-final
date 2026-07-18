"""
Test de la regla determinística de classification (código ICD-10),
validada contra las 30 entradas reales del ground truth. Sin LLM, sin
Stable Diffusion -- corre siempre, instantáneo.
"""

from oncobridge import config
from oncobridge.component2.pipeline import classify_from_icd10
from oncobridge.ingestion.gt_loader import load_ground_truth_base


def test_classify_from_icd10_covers_all_30_real_gt_entries_without_error():
    entries = load_ground_truth_base(config.GT_BASE_DIR)
    assert len(entries) == 30

    for entry in entries:
        classification = classify_from_icd10(entry.icd_10)
        assert classification in ("sospechoso", "benigno", "no_concluyente")


def test_malignant_c_code_is_sospechoso():
    assert classify_from_icd10("C64.9") == "sospechoso"
    assert classify_from_icd10("C34.1") == "sospechoso"


def test_benign_d_code_in_00_36_range_is_benigno():
    assert classify_from_icd10("D30.0") == "benigno"


def test_uncertain_behavior_d_code_in_37_48_range_is_no_concluyente():
    assert classify_from_icd10("D38.1") == "no_concluyente"


def test_unknown_prefix_raises():
    import pytest

    with pytest.raises(ValueError):
        classify_from_icd10("J18.1")
