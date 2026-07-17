"""
Comprime una entrada GT candidata (recuperada por el retriever) al mínimo
de campos necesarios para que el LLM razone si matchea o no con el
paciente. Esta es una pieza central de la estrategia de eficiencia de
contexto. 

Campos que SE excluyen deliberadamente (y por qué):
- meddiffusion_prompt / meddiffusion_negative_prompt / image_generation_notes:
  solo se necesitan para construir el output final, no para que el LLM
  decida si el paciente matchea esta condición.
- views_recommended / positioning_notes: son detalle operativo para el
  radiólogo, no aportan a la decisión de match diagnóstico.
"""

from oncobridge.schemas.ground_truth import GroundTruthEntry


def compress_for_reasoning(entry: GroundTruthEntry) -> dict:
    """
    Devuelve un dict liviano con solo los campos clínicamente relevantes
    para decidir si esta entrada matchea al paciente.
    """
    return {
        "gt_id": entry.gt_id,
        "icd_10": entry.icd_10,
        "icd_10_description": entry.icd_10_description,
        "biomarkers": entry.objective_data.biomarkers,
        "clinical_findings": entry.objective_data.clinical_findings,
        "risk_factors": entry.objective_data.risk_factors,
        "prior_imaging_red_flags": entry.objective_data.prior_imaging_red_flags,
        "symptoms": entry.subjective_data.symptoms,
        "patient_reported_concerns": entry.subjective_data.patient_reported_concerns,
        "onset_pattern": entry.subjective_data.onset_pattern,
        "base_probability": entry.base_probability,
        "urgency_level": entry.urgency_level,
        "notes": entry.notes,
    }