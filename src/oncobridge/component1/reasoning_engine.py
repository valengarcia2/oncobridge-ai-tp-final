"""
Motor de razonamiento de Componente 1. Es la única llamada al LLM que
junta: los datos del paciente, el historial (crudo o resumido),
y los candidatos GT ya comprimidos (recuperados por el retriever). 
Devuelve un resultado "crudo" que el pipeline va a
combinar con la fórmula de probabilidad.
"""

from typing import Literal

from pydantic import BaseModel, Field

from oncobridge import config
from oncobridge.llm.client import LLMClient
from oncobridge.schemas.component1_io import PatientInput, TokenUsage


class RawMatchedHypothesis(BaseModel):
    gt_id: str = Field(description="Debe ser exactamente uno de los gt_id de los candidatos provistos")
    match_probability: float = Field(ge=0.0, le=1.0)
    match_rationale: str


class RawReasoningOutput(BaseModel):
    clinical_summary: str
    reasoning: str
    conclusive: bool = Field(
        description=(
            "true si el sistema pudo concluir algo (incluso 'no es oncológico', "
            "sin necesitar matchear ninguna entrada GT); false solo si los datos "
            "son insuficientes para cualquier conclusión"
        )
    )
    matched_hypotheses: list[RawMatchedHypothesis] = Field(default_factory=list)
    imaging_needed_without_match: bool = Field(
        default=False,
        description=(
            "Solo relevante cuando matched_hypotheses queda vacío y conclusive=true. "
            "true si, aunque ninguna entrada de la base de conocimiento matchea, el "
            "cuadro clínico amerita derivar a imagen igual (hallazgo preocupante sin "
            "diagnóstico específico en la base -- ej. una masa con factores de riesgo "
            "claros pero sin ninguna condición conocida que la explique). false si el "
            "cuadro es benigno/fisiológico y no amerita imagen."
        ),
    )
    no_match_urgency: Literal["alta", "media", "baja"] | None = Field(
        default=None,
        description=(
            "Urgencia clínica estimada SOLO si imaging_needed_without_match es true. "
            "None en cualquier otro caso."
        ),
    )


REASONING_SYSTEM_PROMPT = """\
Sos el motor de análisis clínico de OncoBridge AI, un sistema de apoyo a
la decisión oncológica. NO reemplazás al médico: tu output es una
recomendación que el oncólogo revisa y decide.

Se te da: los datos de un paciente, su historial clínico (posiblemente
resumido), y una lista de entradas candidatas de una base de conocimiento
oncológico (ya pre-seleccionadas por un sistema de recuperación).

Tu tarea:
1. Evaluar, para cada entrada candidata, si el paciente matchea esa
   condición, con una match_probability (0 a 1) y un match_rationale que
   explique la razón clínica (sin inventar datos que el paciente no tiene).
2. Solo incluir en matched_hypotheses las candidatas con evidencia clínica
   real de respaldo — no incluyas una candidata solo porque estaba en la
   lista si los datos del paciente no la sostienen.
3. Determinar `conclusive`: puede ser true incluso si matched_hypotheses
   queda vacío — por ejemplo, si los síntomas claramente indican un cuadro
   benigno o fisiológico que no corresponde a ninguna de las candidatas
   oncológicas provistas. En ese caso, dejá matched_hypotheses vacío pero
   explicá en `reasoning` por qué concluís que no es oncológico.
   `conclusive` debe ser false SOLO cuando los datos son insuficientes
   para cualquier conclusión razonable.
4. NUNCA falsees certeza: si la evidencia es débil, reflejalo con una
   match_probability baja, no la omitas ni la inflés.
5. Si matched_hypotheses queda vacío pero conclusive=true, evaluá
   explícitamente si el cuadro amerita derivar a imagen IGUAL (hallazgo
   preocupante que no coincide con ninguna condición conocida de la base)
   o si es genuinamente benigno/fisiológico. Reflejalo en
   `imaging_needed_without_match` y, si es true, estimá `no_match_urgency`.
   No dejes esto en false por default sin pensarlo: es la diferencia entre
   dos escenarios clínicamente muy distintos.

No tomes ninguna decisión de derivación a imagen ni de urgencia — eso lo
calcula un componente separado a partir de tus probabilidades.
"""


def _build_prompt(
    patient: PatientInput,
    history_text: str,
    compressed_candidates: list[dict],
) -> str:
    labs_text = ", ".join(f"{k}: {v}" for k, v in patient.current_labs.items()) or "sin datos de laboratorio"
    symptoms_text = "; ".join(patient.current_symptoms) or "sin síntomas reportados"
    family_history_text = ", ".join(patient.demographics.family_history) or "sin antecedentes familiares relevantes"

    candidates_text = "SIN CANDIDATOS RECUPERADOS" if not compressed_candidates else "\n\n".join(
        f"--- Candidato {c['gt_id']} ---\n"
        f"Diagnóstico: {c['icd_10_description']}\n"
        f"Biomarcadores esperados: {c['biomarkers']}\n"
        f"Hallazgos clínicos esperados: {c['clinical_findings']}\n"
        f"Factores de riesgo: {c['risk_factors']}\n"
        f"Síntomas típicos: {c['symptoms']}\n"
        f"Frases de paciente típicas: {c['patient_reported_concerns']}\n"
        f"Patrón de evolución típico: {c['onset_pattern']}\n"
        f"Probabilidad base poblacional: {c['base_probability']}\n"
        f"Urgencia si se confirma: {c['urgency_level']}\n"
        f"Notas clínicas: {c['notes']}"
        for c in compressed_candidates
    )

    return f"""\
DATOS DEL PACIENTE
Edad: {patient.demographics.age}, Sexo: {patient.demographics.sex}
Antecedentes familiares: {family_history_text}
Síntomas actuales: {symptoms_text}
Labs actuales: {labs_text}
Historial clínico: {history_text}

CANDIDATOS DE LA BASE DE CONOCIMIENTO
{candidates_text}

Evaluá el caso siguiendo las instrucciones del sistema.
"""


def reason(
    patient: PatientInput,
    history_text: str,
    compressed_candidates: list[dict],
    llm_client: LLMClient | None = None,
    model: str = config.LLM_MODEL_REASONING,
) -> tuple[RawReasoningOutput, TokenUsage]:
    client = llm_client or LLMClient()
    prompt = _build_prompt(patient, history_text, compressed_candidates)

    parsed, token_usage = client.complete_structured(
        prompt=prompt,
        response_schema=RawReasoningOutput,
        model=model,
        system_instruction=REASONING_SYSTEM_PROMPT,
    )
    return parsed, token_usage