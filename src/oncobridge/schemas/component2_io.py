"""
Schema de input/output del Componente 2 
"""

from typing import Literal
from pydantic import BaseModel, Field

from oncobridge.schemas.component1_io import Component1Output, TokenUsage


class ImagingStudy(BaseModel):
    modality: str
    view: str
    image_path: str
    acquisition_date: str


class Component2Input(BaseModel):
    component_1_output: Component1Output
    imaging_study: ImagingStudy


class RegionOfInterest(BaseModel):
    id: str
    location: str
    size_mm: float
    shape: str
    margins: str
    density: str


class Segmentation(BaseModel):
    regions_of_interest: list[RegionOfInterest] = Field(default_factory=list)


Classification = Literal["sospechoso", "benigno", "no_concluyente"]


class Component2Output(BaseModel):
    patient_id: str
    segmentation: Segmentation
    findings: str
    classification: Classification
    confidence: float = Field(ge=0.0, le=1.0)
    final_recommendation: str
    next_steps: list[str] = Field(default_factory=list)
    token_usage: TokenUsage