"""
Test de validación de schemas contra el dataset REAL completo.
"""

import json
import pytest

from oncobridge import config
from oncobridge.schemas.ground_truth import GroundTruthEntry
from oncobridge.schemas.component1_io import PatientInput
from oncobridge.schemas.evaluation import ExpectedOutput


def _gt_files():
    return sorted(config.GT_BASE_DIR.glob("*.json"))


def _case_dirs():
    return sorted(config.CLINICAL_CASES_DIR.glob("case_*"))


def test_dataset_present():
    assert config.GT_BASE_DIR.exists(), f"No se encontró {config.GT_BASE_DIR}"
    assert config.CLINICAL_CASES_DIR.exists(), f"No se encontró {config.CLINICAL_CASES_DIR}"


def test_ground_truth_count_and_ids_unique():
    files = _gt_files()
    assert len(files) == 30, f"Se esperaban 30 entradas GT, se encontraron {len(files)}"
    ids = [json.loads(f.read_text())["gt_id"] for f in files]
    assert len(ids) == len(set(ids)), "Hay gt_id duplicados en la base"


@pytest.mark.parametrize("gt_file", _gt_files(), ids=lambda f: f.stem)
def test_ground_truth_entry_matches_schema(gt_file):
    raw = json.loads(gt_file.read_text())
    entry = GroundTruthEntry.model_validate(raw)
    assert entry.gt_id == gt_file.stem


def test_clinical_cases_count():
    dirs = _case_dirs()
    assert len(dirs) == 110, f"Se esperaban 110 casos clínicos, se encontraron {len(dirs)}"


@pytest.mark.parametrize("case_dir", _case_dirs(), ids=lambda d: d.name)
def test_input_matches_schema(case_dir):
    raw = json.loads((case_dir / "input.json").read_text())
    patient = PatientInput.model_validate(raw)
    assert patient.patient_id


@pytest.mark.parametrize("case_dir", _case_dirs(), ids=lambda d: d.name)
def test_expected_output_matches_schema(case_dir):
    raw = json.loads((case_dir / "expected_output.json").read_text())
    expected = ExpectedOutput.model_validate(raw)
    assert expected.case_id == case_dir.name


def test_expected_output_gt_references_are_valid():
    valid_ids = {json.loads(f.read_text())["gt_id"] for f in _gt_files()}
    broken = []
    for case_dir in _case_dirs():
        exp = json.loads((case_dir / "expected_output.json").read_text())
        for gid in exp.get("correct_gt_ids", []) + exp.get("acceptable_secondary_gt_ids", []):
            if gid not in valid_ids:
                broken.append((case_dir.name, gid))
    assert not broken, f"Referencias a gt_id inexistentes: {broken}"


def test_conclusive_true_without_matches_is_allowed_by_schema():
    found_case = False
    for case_dir in _case_dirs():
        exp_raw = json.loads((case_dir / "expected_output.json").read_text())
        if exp_raw.get("conclusive_ground_truth") is True and not exp_raw.get("correct_gt_ids"):
            expected = ExpectedOutput.model_validate(exp_raw)
            assert expected.conclusive_ground_truth is True
            assert expected.correct_gt_ids == []
            found_case = True
    assert found_case, "No se encontró ningún caso benigno-fisiológico sin GT (se esperaban 6)"