"""
Verifica la CALIDAD del retriever de forma aislada, sin invocar al LLM:
para varios casos reales del dataset, el gt_id correcto tiene que aparecer
entre los top-k candidatos recuperados.
"""

import json
import pytest

from oncobridge import config
from oncobridge.ingestion.gt_loader import load_ground_truth_base
from oncobridge.rag.retriever import HybridRetriever
from oncobridge.schemas.component1_io import PatientInput


@pytest.fixture(scope="module")
def retriever():
    entries = load_ground_truth_base(config.GT_BASE_DIR)
    r = HybridRetriever()
    r.build(entries)
    return r


SAMPLE_CASES = ["case_001", "case_002", "case_003", "case_010", "case_020"]


@pytest.mark.parametrize("case_id", SAMPLE_CASES)
def test_correct_gt_is_in_top_k(retriever, case_id):
    case_dir = config.CLINICAL_CASES_DIR / case_id
    input_raw = json.loads((case_dir / "input.json").read_text(encoding="utf-8"))
    expected_raw = json.loads((case_dir / "expected_output.json").read_text(encoding="utf-8"))

    correct_ids = expected_raw.get("correct_gt_ids", [])
    if not correct_ids:
        pytest.skip(f"{case_id} no tiene correct_gt_ids")

    patient = PatientInput.model_validate(input_raw)
    results = retriever.retrieve(patient, top_k=5)
    retrieved_ids = [gt_id for gt_id, score in results]

    assert any(cid in retrieved_ids for cid in correct_ids), (
        f"{case_id}: se esperaba alguno de {correct_ids} en el top-5, "
        f"se recuperó {retrieved_ids}"
    )