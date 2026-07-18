"""
Pipeline completo de Componente 2. No recibe ningún estudio de imagen real
(no existe en el dataset, acordado así con la cátedra): toma el output de
Componente 1, genera una imagen de referencia ilustrativa para la hipótesis
de mayor match_probability, y arma el Component2Output.

classification sale de una regla determinística sobre el código ICD-10
(C = maligno -> "sospechoso"; D00-D36 = benigno; D37-D48 = comportamiento
incierto -> "no_concluyente"), validada contra las 30 entradas reales del
ground truth. confidence es directamente el match_probability que ya
calculó C1 -- C2 no inventa un número nuevo, empaqueta y presenta
visualmente lo que C1 ya concluyó.

La imagen generada NO viaja en Component2Output (el contrato no se toca):
se cachea en disco por gt_id y cualquier consumidor (la UI) la ubica con
image_synthesizer.reference_image_path(top_hypothesis.gt_id).
"""

from oncobridge import config
from oncobridge.component2.image_synthesizer import generate_reference_image
from oncobridge.component2.reasoning_engine import structure_findings
from oncobridge.ingestion.gt_loader import load_ground_truth_base
from oncobridge.llm.client import LLMClient
from oncobridge.schemas.component1_io import Component1Output
from oncobridge.schemas.component2_io import Classification, Component2Output
from oncobridge.schemas.ground_truth import GroundTruthEntry


def classify_from_icd10(icd_10: str) -> Classification:
    """C = maligno -> sospechoso; D00-D36 = benigno; D37-D48 = no_concluyente."""
    letter = icd_10[0]
    if letter == "C":
        return "sospechoso"
    if letter == "D":
        number = int(icd_10[1:3])
        if 0 <= number <= 36:
            return "benigno"
        if 37 <= number <= 48:
            return "no_concluyente"
    raise ValueError(f"Código ICD-10 inesperado, no matchea la regla conocida: {icd_10}")


def run_component2(
    c1_output: Component1Output,
    gt_entries: list[GroundTruthEntry] | None = None,
    llm_client: LLMClient | None = None,
) -> Component2Output:
    if not c1_output.matched_ground_truths:
        raise ValueError("Componente 2 requiere al menos una hipótesis matcheada por Componente 1")

    top_hypothesis = c1_output.matched_ground_truths[0]

    if gt_entries is None:
        gt_entries = load_ground_truth_base(config.GT_BASE_DIR)
    entry = next(e for e in gt_entries if e.gt_id == top_hypothesis.gt_id)

    generate_reference_image(entry)

    client = llm_client or LLMClient()
    structured, token_usage = structure_findings(top_hypothesis, llm_client=client)

    return Component2Output(
        patient_id=c1_output.patient_id,
        segmentation=structured.segmentation,
        findings=structured.findings,
        classification=classify_from_icd10(top_hypothesis.icd_10),
        confidence=top_hypothesis.match_probability,
        final_recommendation=structured.final_recommendation,
        next_steps=structured.next_steps,
        token_usage=token_usage,
    )
