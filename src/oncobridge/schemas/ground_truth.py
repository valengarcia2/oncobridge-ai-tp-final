"""
Schema de la base de Ground Truth oncológico 
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field


class Biomarkers(BaseModel):
    model_config = {"extra": "allow"}


class ObjectiveData(BaseModel):
    biomarkers: dict[str, str] = Field(default_factory=dict)
    clinical_findings: list[str] = Field(default_factory=list)
    risk_factors: list[str] = Field(default_factory=list)
    prior_imaging_red_flags: list[str] = Field(default_factory=list)


class SubjectiveData(BaseModel):
    symptoms: list[str] = Field(default_factory=list)
    patient_reported_concerns: list[str] = Field(default_factory=list)
    onset_pattern: str = ""


class ImagingLocation(BaseModel):
    body_region: str = ""
    anatomical_landmarks: str = ""
    bilateral_comparison_required: bool = False
    priority_zones: list[str] = Field(default_factory=list)
    positioning_notes: str = ""


class RadiologistGuidance(BaseModel):
    modality_priority: list[str] = Field(default_factory=list)
    views_recommended: list[str] = Field(default_factory=list)
    imaging_location: ImagingLocation
    expected_imaging_findings: str = ""
    meddiffusion_prompt: str = ""
    meddiffusion_negative_prompt: str = ""
    image_generation_notes: str = ""


class GroundTruthMeta(BaseModel):
    model_config = {"extra": "allow"}
    category: Optional[str] = None
    organ: Optional[str] = None


class GroundTruthEntry(BaseModel):
    gt_id: str
    icd_10: str
    icd_10_description: str
    objective_data: ObjectiveData
    subjective_data: SubjectiveData
    radiologist_guidance: RadiologistGuidance
    base_probability: float = Field(ge=0.0, le=1.0)
    urgency_level: Literal["alta", "media", "baja"]
    notes: str = ""
    meta: Optional[GroundTruthMeta] = Field(default=None, alias="_meta")

    model_config = {"populate_by_name": True}