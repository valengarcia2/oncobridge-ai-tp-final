"""
Test del motor de razonamiento de C2 (estructura texto ya existente del GT,
no inventa contenido nuevo). Requiere GEMINI_API_KEY real, se salta si no
está configurada -- mismo patrón que los tests de C1.
"""

import os

import pytest

from oncobridge import config
from oncobridge.component1.pipeline import run_component1
from oncobridge.component2.reasoning_engine import structure_findings
from oncobridge.ingestion.gt_loader import load_ground_truth_base
from oncobridge.rag.retriever import HybridRetriever
from oncobridge.schemas.component1_io import PatientInput


@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="Requiere GEMINI_API_KEY real")
def test_structure_findings_returns_valid_schema_and_token_usage():
    entries = load_ground_truth_base(config.GT_BASE_DIR)
    retriever = HybridRetriever()
    retriever.build(entries)

    raw = (config.CLINICAL_CASES_DIR / "case_002" / "input.json").read_text(encoding="utf-8")
    import json

    patient = PatientInput.model_validate(json.loads(raw))
    c1_output = run_component1(patient, gt_entries=entries, retriever=retriever)
    assert c1_output.matched_ground_truths, "case_002 debería matchear al menos una hipótesis"

    top_hypothesis = c1_output.matched_ground_truths[0]
    structured, token_usage = structure_findings(top_hypothesis)

    assert structured.findings
    assert structured.final_recommendation
    assert token_usage.total_tokens > 0
