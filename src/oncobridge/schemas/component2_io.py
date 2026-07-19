"""
Schema de input/output del Componente 2

Componente 2 no recibe ningún estudio de imagen real del paciente (el
dataset es "clinical-only", acordado así con la cátedra): el input es
únicamente el output de Componente 1. C2 genera una imagen de referencia
ilustrativa a partir de la hipótesis de mayor match_probability y arma el
reporte para el radiólogo a partir del ground truth ya matcheado, sin
comparar contra ningún estudio real.
"""

from typing import Literal
from pydantic import BaseModel, Field

from oncobridge.schemas.component1_io import Component1Output, TokenUsage


class Component2Input(BaseModel):
    component_1_output: Component1Output


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