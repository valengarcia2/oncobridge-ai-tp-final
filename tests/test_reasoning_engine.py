"""
Tests del motor de razonamiento:
- Test de construcción del prompt: no necesita LLM, corre siempre.
- Tests de razonamiento real contra casos reales: requieren GEMINI_API_KEY,
  se saltan automáticamente si no está configurada.
"""

import json
import os
import pytest

from oncobridge import config
from oncobridge.ingestion.gt_loader import load_ground_truth_base
from oncobridge.rag.retriever import HybridRetriever
from oncobridge.rag.compressor import compress_for_reasoning
from oncobridge.component1.history_summarizer import summarize_if_needed
from oncobridge.component1.reasoning_engine import reason, _build_prompt
from oncobridge.schemas.component1_io import PatientInput


def test_build_prompt_includes_patient_and_candidate_data():
    patient = PatientInput.model_validate({
        "patient_id": "TEST-001",
        "demographics": {"age": 55, "sex": "F", "family_history": ["breast_cancer"]},
        "current_symptoms": ["masa palpable"],
        "medical_history": [],
        "current_labs": {"CA_15_3": 40.0},
    })
    candidate = {
        "gt_id": "GT-TEST-001", "icd_10": "C00", "icd_10_description": "Test",
        "biomarkers": {}, "clinical_findings": [], "risk_factors": [],
        "prior_imaging_red_flags": [], "symptoms": [], "patient_reported_concerns": [],
        "onset_pattern": "", "base_probability": 0.5, "urgency_level": "alta", "notes": "",
    }
    prompt = _build_prompt(patient, "sin antecedentes", [candidate])

    assert "masa palpable" in prompt
    assert "GT-TEST-001" in prompt
    assert "CA_15_3" in prompt


def test_build_prompt_handles_empty_candidates():
    patient = PatientInput.model_validate({
        "patient_id": "TEST-002",
        "demographics": {"age": 30, "sex": "M", "family_history": []},
        "current_symptoms": ["dolor leve"],
        "medical_history": [],
        "current_labs": {},
    })
    prompt = _build_prompt(patient, "sin antecedentes", [])
    assert "SIN CANDIDATOS RECUPERADOS" in prompt


def _run_full_pipeline_up_to_reasoning(case_id: str):
    """Junta retriever + compressor + history_summarizer + reasoning."""
    entries = load_ground_truth_base(config.GT_BASE_DIR)
    retriever = HybridRetriever()
    retriever.build(entries)
    entries_by_id = {e.gt_id: e for e in entries}

    case_dir = config.CLINICAL_CASES_DIR / case_id
    input_raw = json.loads((case_dir / "input.json").read_text())
    patient = PatientInput.model_validate(input_raw)

    history_text, _ = summarize_if_needed(patient.medical_history)
    top_candidates = retriever.retrieve(patient, top_k=config.RETRIEVER_TOP_K)
    compressed = [compress_for_reasoning(entries_by_id[gt_id]) for gt_id, _ in top_candidates]

    return reason(patient, history_text, compressed)


@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="Requiere GEMINI_API_KEY real")
def test_reasoning_on_clear_positive_case():
    """case_002: TP claro (GT-RENAL-001) -> debe matchear con alta probabilidad."""
    output, token_usage = _run_full_pipeline_up_to_reasoning("case_002")

    assert token_usage.total_tokens > 0
    assert output.conclusive is True
    matched_ids = [h.gt_id for h in output.matched_hypotheses]
    assert "GT-RENAL-001" in matched_ids


@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="Requiere GEMINI_API_KEY real")
def test_reasoning_on_benign_physiological_case_without_gt():
    """case_049: benigno-fisiológico, correct_gt_ids=[] pero conclusive=true."""
    output, token_usage = _run_full_pipeline_up_to_reasoning("case_049")

    assert token_usage.total_tokens > 0
    assert output.conclusive is True
    if output.matched_hypotheses:
        assert all(h.match_probability < 0.5 for h in output.matched_hypotheses)