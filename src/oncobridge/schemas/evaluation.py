"""
Schema del expected_output.json de cada caso clínico
"""

from typing import Literal
from pydantic import BaseModel, Field

from oncobridge.schemas.component1_io import Recommendation, Urgency

Difficulty = Literal["facil", "moderado", "dificil"]


class ExpectedOutput(BaseModel):
    case_id: str
    correct_gt_ids: list[str] = Field(default_factory=list)
    acceptable_secondary_gt_ids: list[str] = Field(default_factory=list)
    imaging_needed_ground_truth: bool
    urgency_ground_truth: Urgency
    specialist_decision: Recommendation
    conclusive_ground_truth: bool
    difficulty: Difficulty
    notes: str = ""