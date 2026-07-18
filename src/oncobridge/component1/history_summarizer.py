"""
Resumen progresivo CONDICIONAL del historial clínico. Para la mayoría de los casos (91 de 110, con 1-4 eventos de
historial) esta función es un NO-OP — no gasta ninguna llamada extra al
LLM. Solo para los ~19 casos "COMPLEX" (8-11 eventos) se invoca al LLM
para resumir.
"""

from pydantic import BaseModel, Field

from oncobridge import config
from oncobridge.llm.client import LLMClient
from oncobridge.schemas.component1_io import MedicalHistoryEvent, TokenUsage


class HistorySummary(BaseModel):
    summary: str = Field(
        description="Resumen clínico priorizando eventos oncológicos-relevantes y recientes"
    )


def _format_raw_history(events: list[MedicalHistoryEvent]) -> str:
    """Formato de texto simple para el caso no-COMPLEX (sin LLM)."""
    if not events:
        return "Sin antecedentes registrados."
    return "; ".join(f"{e.date}: {e.event}" for e in events)


SUMMARIZATION_SYSTEM_PROMPT = """\
Sos un asistente que resume historiales clínicos oncológicos para un
oncólogo, priorizando:
1. Eventos relacionados a patología oncológica o estudios de imagen previos.
2. Eventos más recientes por sobre los más antiguos.
3. Cualquier resultado con hallazgos sospechosos (BI-RADS alto, biopsias
   con resultado atípico, etc.).
Excluí detalles administrativos irrelevantes. Sé conciso: 2-4 oraciones.
"""


def summarize_if_needed(
    events: list[MedicalHistoryEvent],
    llm_client: LLMClient | None = None,
    threshold: int = config.COMPLEX_HISTORY_THRESHOLD,
    model: str = config.LLM_MODEL_SUMMARIZATION,
) -> tuple[str, TokenUsage | None]:
    """
    Devuelve (texto_de_historial, token_usage_o_None).
    - Si len(events) <= threshold: no-op, historial crudo formateado,
      token_usage=None (no se llamó al LLM).
    - Si len(events) > threshold: invoca al LLM (modelo económico) y
      devuelve el resumen junto con su token_usage real.
    """
    if len(events) <= threshold:
        return _format_raw_history(events), None

    client = llm_client or LLMClient()
    raw_text = _format_raw_history(events)
    prompt = f"Historial clínico completo:\n{raw_text}\n\nResumilo."

    parsed, token_usage = client.complete_structured(
        prompt=prompt,
        response_schema=HistorySummary,
        model=model,
        system_instruction=SUMMARIZATION_SYSTEM_PROMPT,
    )
    return parsed.summary, token_usage