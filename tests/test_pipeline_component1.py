"""
Test del pipeline completo de Componente 1 de punta a punta.
Requiere GEMINI_API_KEY real.
"""

import json
import os
import pytest

from oncobridge import config
from oncobridge.ingestion.gt_loader import load_ground_truth_base
from oncobridge.rag.retriever import HybridRetriever
from oncobridge.component1.pipeline import run_component1
from oncobridge.schemas.component1_io import PatientInput


@pytest.fixture(scope="module")
def gt_and_retriever():
    entries = load_ground_truth_base(config.GT_BASE_DIR)
    retriever = HybridRetriever()
    retriever.build(entries)
    return entries, retriever


def _load_patient(case_id: str) -> PatientInput:
    raw = json.loads((config.CLINICAL_CASES_DIR / case_id / "input.json").read_text())
    return PatientInput.model_validate(raw)


@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="Requiere GEMINI_API_KEY real")
def test_pipeline_positive_case_end_to_end(gt_and_retriever):
    """case_002: TP claro -> debe recomendar derivación con hipótesis matcheada."""
    entries, retriever = gt_and_retriever
    patient = _load_patient("case_002")

    output = run_component1(patient, gt_entries=entries, retriever=retriever)

    assert output.patient_id == patient.patient_id
    assert output.recommendation == "DERIVAR_A_IMAGEN"
    assert output.conclusive is True
    assert len(output.matched_ground_truths) > 0
    assert output.token_usage.total_tokens > 0
    assert 0 < output.token_usage.retrieved_gt_entries <= config.RETRIEVER_TOP_K

    top_hypothesis = output.matched_ground_truths[0]
    assert top_hypothesis.radiologist_instructions.meddiffusion_reference_prompt != ""


@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="Requiere GEMINI_API_KEY real")
def test_pipeline_benign_physiological_case_end_to_end(gt_and_retriever):
    """case_049: benigno-fisiológico -> conclusive=true, sin necesitar GT."""
    entries, retriever = gt_and_retriever
    patient = _load_patient("case_049")

    output = run_component1(patient, gt_entries=entries, retriever=retriever)

    assert output.conclusive is True
    assert output.recommendation in ("NO_DERIVAR", "SEGUIMIENTO_CLINICO")


@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="Requiere GEMINI_API_KEY real")
def test_pipeline_healthy_case_returns_sin_elementos(gt_and_retriever):
    """Un caso 'sano' real del dataset."""
    entries, retriever = gt_and_retriever

    healthy_case_id = None
    for case_dir in sorted(config.CLINICAL_CASES_DIR.glob("case_*")):
        expected = json.loads((case_dir / "expected_output.json").read_text())
        if not expected["correct_gt_ids"] and expected["conclusive_ground_truth"] is False:
            healthy_case_id = case_dir.name
            break
    assert healthy_case_id is not None, "No se encontró ningún caso sano en el dataset"

    patient = _load_patient(healthy_case_id)
    output = run_component1(patient, gt_entries=entries, retriever=retriever)

    assert output.recommendation in ("SIN_ELEMENTOS_PARA_EVALUAR", "NO_DERIVAR")