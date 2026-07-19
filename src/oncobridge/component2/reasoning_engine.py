"""
Única llamada LLM de Componente 2: estructura (no inventa) la información
clínica que ya viene en el ground truth matcheado -- priority_zones,
landmarks anatómicos, hallazgos esperados -- en el formato del contrato de
salida (segmentation, findings, final_recommendation, next_steps).

classification y confidence NO pasan por acá: se calculan de forma
determinística en pipeline.py (regla de ICD-10 y el match_probability que
ya calculó C1), para no dejar que el LLM contradiga un dato del que ya
tenemos certeza.
"""

from pydantic import BaseModel, Field

from oncobridge import config
from oncobridge.llm.client import LLMClient
from oncobridge.schemas.component1_io import MatchedHypothesis, TokenUsage
from oncobridge.schemas.component2_io import Segmentation

SYSTEM_INSTRUCTION = """\
Sos un asistente que redacta, para un especialista en imágenes, qué debería \
esperar ver en el estudio dado un diagnóstico ya identificado por el \
sistema. No diagnostiques ni inventes hallazgos nuevos: estructurá y \
redactá en lenguaje clínico claro la información que se te da, sin agregar \
datos que no estén en el contexto provisto.\
"""


class C2StructuredFindings(BaseModel):
    segmentation: Segmentation
    findings: str = Field(description="Hallazgos esperados, en lenguaje clínico para el radiólogo")
    final_recommendation: str
    next_steps: list[str] = Field(default_factory=list)


def structure_findings(
    hypothesis: MatchedHypothesis,
    expected_imaging_findings: str,
    llm_client: LLMClient | None = None,
    model: str = config.LLM_MODEL_VISION,
) -> tuple[C2StructuredFindings, TokenUsage]:
    client = llm_client or LLMClient()
    ri = hypothesis.radiologist_instructions
    prompt = (
        f"Diagnóstico identificado: {hypothesis.icd_10_description} ({hypothesis.icd_10}).\n"
        f"Zona anatómica: {ri.imaging_location.body_region} — "
        f"{ri.imaging_location.anatomical_landmarks}.\n"
        f"Zonas prioritarias: {', '.join(ri.imaging_location.priority_zones)}.\n"
        f"Contexto clínico del match: {ri.clinical_context_for_radiologist}.\n"
        f"Hallazgos esperados en la imagen: {expected_imaging_findings}.\n\n"
        "Armá la segmentación (regiones de interés esperadas), los hallazgos, "
        "la recomendación final y los próximos pasos sugeridos. Si el tamaño, "
        "forma o márgenes exactos no están especificados en el texto, estimalos "
        "de forma ilustrativa y razonable para este tipo de hallazgo, dejándolo "
        "claro como una referencia orientativa y no una medición real."
    )
    return client.complete_structured(
        prompt=prompt,
        response_schema=C2StructuredFindings,
        model=model,
        system_instruction=SYSTEM_INSTRUCTION,
    )
