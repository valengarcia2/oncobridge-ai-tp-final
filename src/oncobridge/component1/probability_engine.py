"""
Motor de probabilidad de Componente 1: calcula `imaging_needed_probability`,
`recommendation` y `urgency` de forma DETERMINÍSTICA (sin LLM), a partir de
las hipótesis ya evaluadas por el reasoning_engine.

Fórmula:
    imaging_needed_probability = max_i( match_probability_i * urgency_weight_i )

Se toma el MÁXIMO entre hipótesis, no el promedio: una sola hipótesis
urgente y probable debería alcanzar para recomendar derivación, aunque
haya otras hipótesis secundarias de baja probabilidad.
"""

from dataclasses import dataclass

from oncobridge import config
from oncobridge.schemas.component1_io import Recommendation, Urgency


@dataclass
class HypothesisForProbability:
    """
    Insumo mínimo que necesita este motor por cada hipótesis: su
    match_probability (del reasoning_engine) y su urgency_level (del GT
    original — el pipeline del Paso 9 es quien junta ambas cosas).
    """
    match_probability: float
    urgency_level: str  # "alta" | "media" | "baja"


def compute_imaging_needed_probability(
    hypotheses: list[HypothesisForProbability],
    conclusive: bool,
) -> tuple[float, Recommendation, Urgency]:
    """
    Devuelve (imaging_needed_probability, recommendation, urgency).

    1. Sin hipótesis + conclusive=False -> SIN_ELEMENTOS_PARA_EVALUAR.
    2. Sin hipótesis + conclusive=True -> NO_DERIVAR, probabilidad 0
       (ej. cuadro benigno-fisiológico).
    3. Con hipótesis -> se aplica la fórmula y los umbrales configurados.
    """
    if not hypotheses:
        if conclusive:
            return 0.0, "NO_DERIVAR", "ninguna"
        return 0.0, "SIN_ELEMENTOS_PARA_EVALUAR", "ninguna"

    weighted_scores = [
        (h.match_probability * config.URGENCY_WEIGHTS[h.urgency_level], h.urgency_level)
        for h in hypotheses
    ]
    imaging_needed_probability, driving_urgency_level = max(weighted_scores, key=lambda x: x[0])

    if imaging_needed_probability >= config.DERIVAR_THRESHOLD:
        recommendation: Recommendation = "DERIVAR_A_IMAGEN"
        urgency: Urgency = driving_urgency_level
    elif imaging_needed_probability >= config.SEGUIMIENTO_THRESHOLD:
        recommendation = "SEGUIMIENTO_CLINICO"
        urgency = "baja"
    else:
        recommendation = "NO_DERIVAR"
        urgency = "ninguna"

    return round(imaging_needed_probability, 4), recommendation, urgency