"""
Schema de input/output del Componente 1 
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field

from oncobridge.schemas.ground_truth import ImagingLocation


class Demographics(BaseModel):
    age: int
    sex: Literal["F", "M"]
    family_history: list[str] = Field(default_factory=list)


class MedicalHistoryEvent(BaseModel):
    date: str
    event: str


class PatientInput(BaseModel):
    patient_id: str
    demographics: Demographics
    current_symptoms: list[str] = Field(default_factory=list)
    medical_history: list[MedicalHistoryEvent] = Field(default_factory=list)
    current_labs: dict[str, object] = Field(default_factory=dict)


class RadiologistInstructions(BaseModel):
    suggested_modalities: list[str] = Field(default_factory=list)
    imaging_location: ImagingLocation
    clinical_context_for_radiologist: str = ""
    meddiffusion_reference_prompt: str = ""
    meddiffusion_negative_prompt: str = ""
    reference_images_note: str = ""


class MatchedHypothesis(BaseModel):
    gt_id: str
    icd_10: str
    icd_10_description: str
    match_probability: float = Field(ge=0.0, le=1.0)
    match_rationale: str
    radiologist_instructions: RadiologistInstructions


class TokenUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str
    retrieved_gt_entries: Optional[int] = None
    gt_entries_in_context: Optional[int] = None


Recommendation = Literal[
    "DERIVAR_A_IMAGEN", "NO_DERIVAR", "SEGUIMIENTO_CLINICO", "SIN_ELEMENTOS_PARA_EVALUAR"
]
Urgency = Literal["alta", "media", "baja", "ninguna"]


class Component1Output(BaseModel):
    patient_id: str
    clinical_summary: str
    matched_ground_truths: list[MatchedHypothesis] = Field(default_factory=list)
    imaging_needed_probability: float = Field(ge=0.0, le=1.0)
    reasoning: str
    recommendation: Recommendation
    urgency: Urgency
    conclusive: bool
    token_usage: TokenUsage