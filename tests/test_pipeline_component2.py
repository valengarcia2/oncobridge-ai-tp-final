"""
Test del pipeline completo de Componente 2, de punta a punta. Requiere
GEMINI_API_KEY real. La primera vez que corre también descarga/carga
Stable Diffusion para generar la imagen de referencia (puede tardar) --
correr antes scripts/generate_reference_images.py para tener el cache
tibio y que este test sea rápido.
"""

import json
import os

import pytest

from oncobridge import config
from oncobridge.component1.pipeline import run_component1
from oncobridge.component2.image_synthesizer import reference_image_path
from oncobridge.component2.pipeline import run_component2
from oncobridge.ingestion.gt_loader import load_ground_truth_base
from oncobridge.rag.retriever import HybridRetriever
from oncobridge.schemas.component1_io import PatientInput


@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="Requiere GEMINI_API_KEY real")
def test_pipeline_component2_end_to_end_for_positive_case():
    entries = load_ground_truth_base(config.GT_BASE_DIR)
    retriever = HybridRetriever()
    retriever.build(entries)

    raw = json.loads((config.CLINICAL_CASES_DIR / "case_002" / "input.json").read_text())
    patient = PatientInput.model_validate(raw)

    c1_output = run_component1(patient, gt_entries=entries, retriever=retriever)
    assert c1_output.matched_ground_truths

    c2_output = run_component2(c1_output, gt_entries=entries)

    assert c2_output.patient_id == c1_output.patient_id
    assert c2_output.classification in ("sospechoso", "benigno", "no_concluyente")
    assert c2_output.confidence == c1_output.matched_ground_truths[0].match_probability
    assert c2_output.findings
    assert c2_output.token_usage.total_tokens > 0

    top_gt_id = c1_output.matched_ground_truths[0].gt_id
    assert reference_image_path(top_gt_id).exists()
