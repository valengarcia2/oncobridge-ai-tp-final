"""
Tests del resumen de historial. Separamos deliberadamente:
- Tests del camino NO-OP (sin LLM): corren siempre, sin necesitar API key.
- Test del camino COMPLEX (con LLM real): requiere GEMINI_API_KEY válida,
  se salta automáticamente si no está configurada.
"""

import json
import os
import pytest

from oncobridge import config
from oncobridge.schemas.component1_io import MedicalHistoryEvent
from oncobridge.component1.history_summarizer import summarize_if_needed


def _events_from_case(case_id: str) -> list[MedicalHistoryEvent]:
    raw = json.loads((config.CLINICAL_CASES_DIR / case_id / "input.json").read_text())
    return [MedicalHistoryEvent(**e) for e in raw["medical_history"]]


def test_short_history_is_noop_no_token_usage():
    """case_001 tiene pocos eventos (< threshold) -> no debe llamar al LLM."""
    events = _events_from_case("case_001")
    assert len(events) <= config.COMPLEX_HISTORY_THRESHOLD

    text, token_usage = summarize_if_needed(events)

    assert token_usage is None
    assert isinstance(text, str) and len(text) > 0


def test_empty_history_returns_placeholder_text():
    text, token_usage = summarize_if_needed([])
    assert token_usage is None
    assert "sin antecedentes" in text.lower()


@pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY"),
    reason="Requiere GEMINI_API_KEY real para probar el resumen vía LLM",
)
def test_complex_history_gets_summarized_by_llm():
    """case_091 es uno de los casos COMPLEX reales (10 eventos)."""
    events = _events_from_case("case_091")
    assert len(events) > config.COMPLEX_HISTORY_THRESHOLD

    text, token_usage = summarize_if_needed(events)

    assert token_usage is not None
    assert token_usage.total_tokens > 0
    assert isinstance(text, str) and len(text) > 0
    raw_length = sum(len(e.event) for e in events)
    assert len(text) < raw_length