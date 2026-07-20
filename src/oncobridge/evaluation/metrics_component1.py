"""
Métricas de evaluación de Componente 1, calculadas a
partir de pares (Component1Output real, ExpectedOutput del dataset). Sin
LLM: son funciones puras sobre los outputs ya generados.

Precisión de GT Match: la consigna la define como "% de casos donde el
gt_id de mayor probabilidad coincide con el diagnóstico correcto" -- eso
solo tiene sentido cuando existe un `correct_gt_ids` real para comparar.
En el dataset real hay 6 de 110 casos "benigno-fisiológico" con
correct_gt_ids vacío (viral autolimitado, adenopatía reactiva, pólipo
hiperplásico, angiomiolipoma conocido, etc.), y de esos, case_109 además
pide DERIVAR_A_IMAGEN sin que la base GT cubra la condición (sarcoma
retroperitoneal). gt_match_is_applicable() excluye esos casos del
denominador de esta métrica puntual -- no se premian ni se castigan,
documentado así en el README. Es una regla derivada de los datos
(correct_gt_ids vacío o no), no un `if case_id == "case_109"` hardcodeado.

El resto de las métricas (accuracy de derivación, sensibilidad,
especificidad, conclusive) sí se calculan sobre los 110 casos, porque no
dependen de que exista un gt_id de referencia.
"""

from dataclasses import dataclass, field

import numpy as np

from oncobridge.schemas.component1_io import Component1Output
from oncobridge.schemas.evaluation import ExpectedOutput


def gt_match_is_applicable(expected: ExpectedOutput) -> bool:
    """La métrica de GT match solo se puede evaluar si hay un correct_gt_ids real."""
    return bool(expected.correct_gt_ids)


def gt_match_is_correct(output: Component1Output, expected: ExpectedOutput) -> bool:
    """
    ¿El gt_id de mayor match_probability coincide con correct_gt_ids?
    Llamar solo si gt_match_is_applicable(expected) es True.
    """
    if not output.matched_ground_truths:
        return False
    top_gt_id = output.matched_ground_truths[0].gt_id
    return top_gt_id in expected.correct_gt_ids


def imaging_decision_is_correct(output: Component1Output, expected: ExpectedOutput) -> bool:
    """¿La recomendación final (de las 4 categorías) coincide con la decisión esperada?"""
    return output.recommendation == expected.specialist_decision


def conclusive_matches_expected(output: Component1Output, expected: ExpectedOutput) -> bool:
    return output.conclusive == expected.conclusive_ground_truth


@dataclass
class ConfusionBucket:
    """Clasificación TP/TN/FP/FN de un solo caso, sobre la decisión binaria 'derivar a imagen'."""

    true_positive: bool = False
    true_negative: bool = False
    false_positive: bool = False
    false_negative: bool = False


def imaging_confusion_bucket(output: Component1Output, expected: ExpectedOutput) -> ConfusionBucket:
    """
    Compara `imaging_needed_ground_truth` (bool) contra si el sistema
    recomendó derivar (recommendation == DERIVAR_A_IMAGEN). Agregando esto
    sobre muchos casos se calculan sensibilidad y especificidad.
    """
    needed = expected.imaging_needed_ground_truth
    derived = output.recommendation == "DERIVAR_A_IMAGEN"

    if needed and derived:
        return ConfusionBucket(true_positive=True)
    if needed and not derived:
        return ConfusionBucket(false_negative=True)
    if not needed and derived:
        return ConfusionBucket(false_positive=True)
    return ConfusionBucket(true_negative=True)


@dataclass
class Component1MetricsReport:
    n_cases: int
    accuracy_derivacion: float
    sensibilidad: float
    especificidad: float
    precision_gt_match: float
    n_casos_gt_match_aplicable: int
    tokens_promedio_por_caso: float
    calibracion_gt_probability: float | None
    detalle_por_caso: list[dict] = field(default_factory=list)


def aggregate_component1_metrics(
    pairs: list[tuple[Component1Output, ExpectedOutput]],
) -> Component1MetricsReport:
    """
    Agrega las métricas sobre una lista de (output real, expected)
    """
    n_cases = len(pairs)
    if n_cases == 0:
        raise ValueError("Se necesita al menos un caso para calcular métricas")

    detalle = []
    decision_hits = 0
    buckets = []
    gt_match_hits = 0
    gt_match_applicable = 0
    total_tokens = 0

    match_probabilities: list[float] = []
    match_correctness: list[int] = []

    for output, expected in pairs:
        decision_correct = imaging_decision_is_correct(output, expected)
        decision_hits += int(decision_correct)

        bucket = imaging_confusion_bucket(output, expected)
        buckets.append(bucket)

        applicable = gt_match_is_applicable(expected)
        gt_correct = gt_match_is_correct(output, expected) if applicable else None
        if applicable:
            gt_match_applicable += 1
            gt_match_hits += int(gt_correct)
            if output.matched_ground_truths:
                match_probabilities.append(output.matched_ground_truths[0].match_probability)
                match_correctness.append(int(gt_correct))

        total_tokens += output.token_usage.total_tokens

        detalle.append(
            {
                "case_id": expected.case_id,
                "decision_correct": decision_correct,
                "gt_match_applicable": applicable,
                "gt_match_correct": gt_correct,
                "conclusive_correct": conclusive_matches_expected(output, expected),
            }
        )

    true_positives = sum(b.true_positive for b in buckets)
    true_negatives = sum(b.true_negative for b in buckets)
    false_positives = sum(b.false_positive for b in buckets)
    false_negatives = sum(b.false_negative for b in buckets)

    sensibilidad = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else float("nan")
    )
    especificidad = (
        true_negatives / (true_negatives + false_positives)
        if (true_negatives + false_positives) > 0
        else float("nan")
    )

    calibracion = None
    if len(match_probabilities) >= 2 and len(set(match_correctness)) > 1:
        calibracion = float(np.corrcoef(match_probabilities, match_correctness)[0, 1])

    return Component1MetricsReport(
        n_cases=n_cases,
        accuracy_derivacion=decision_hits / n_cases,
        sensibilidad=sensibilidad,
        especificidad=especificidad,
        precision_gt_match=(
            gt_match_hits / gt_match_applicable if gt_match_applicable > 0 else float("nan")
        ),
        n_casos_gt_match_aplicable=gt_match_applicable,
        tokens_promedio_por_caso=total_tokens / n_cases,
        calibracion_gt_probability=calibracion,
        detalle_por_caso=detalle,
    )
