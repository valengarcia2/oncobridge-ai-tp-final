"""
Métricas de evaluación de Componente 2 (§5.2 de la consigna, tabla de C2),
calculadas a partir de (Component2Output real, ExpectedOutput, la base GT).
Sin LLM: funciones puras sobre outputs ya generados.

C2 solo corre cuando C1 recomienda DERIVAR_A_IMAGEN con al menos una
hipótesis matcheada -- por diseño, ningún caso NO_DERIVAR llega nunca a
esta evaluación. De los 6 casos sin correct_gt_ids (ver metrics_component1),
solo case_109 (DERIVAR_A_IMAGEN sin match en la base) podría llegar a
invocar C2 -- y, igual que en C1, queda excluido del denominador de las
métricas que necesitan comparar contra un diagnóstico correcto conocido
(no hay ninguno para ese caso). classification_is_applicable() aplica la
misma regla derivada de los datos (correct_gt_ids vacío o no).

No hay imagen real de paciente ni segmentación anotada por píxel (acordado
con la cátedra) -- las métricas de esta sección son proxies honestos, no
la métrica literal de la consigna:

- "Precisión de Segmentación (IoU)": no hay máscaras de píxeles. El proxy
  usado es un IoU de TEXTO: superposición de palabras (Jaccard) entre las
  zonas que C2 reportó (`segmentation.regions_of_interest[].location`) y
  las zonas/landmarks reales de la condición correcta del caso
  (`priority_zones` + `anatomical_landmarks` del GT correcto). Es la misma
  fórmula de intersección-sobre-unión que un IoU de píxeles, aplicada a
  conjuntos de palabras en vez de conjuntos de píxeles.
- "Sensibilidad/Especificidad de Hallazgos": no hay lesiones anotadas en
  una imagen real. El proxy usa `classification` ("sospechoso" = lesión
  detectada) comparado contra si el diagnóstico CORRECTO real del caso
  (`correct_gt_ids`) es maligno o no, vía la misma regla ICD-10 de
  `component2/pipeline.py`.
- "Concordancia Clínica": correlación entre `confidence` (que ya es el
  `match_probability` de C1, no un número nuevo de C2) y si la
  clasificación resultó correcta -- mismo enfoque que la calibración de C1.
- "Calidad del Informe": la consigna la define como evaluación humana
  (completitud/precisión/claridad) -- no automatizable, se documenta como
  tal, igual que "Coherencia del Razonamiento" de C1.
"""

from dataclasses import dataclass, field

import numpy as np

from oncobridge.component2.pipeline import classify_from_icd10
from oncobridge.schemas.component2_io import Classification, Component2Output
from oncobridge.schemas.evaluation import ExpectedOutput
from oncobridge.schemas.ground_truth import GroundTruthEntry


def classification_is_applicable(expected: ExpectedOutput) -> bool:
    """Igual que en C1: solo se puede saber la clasificación correcta si hay correct_gt_ids."""
    return bool(expected.correct_gt_ids)


def expected_classification(
    expected: ExpectedOutput, gt_index: dict[str, GroundTruthEntry]
) -> Classification | None:
    """Clasificación verdadera del caso, derivada del primer correct_gt_ids vía regla ICD-10."""
    if not classification_is_applicable(expected):
        return None
    entry = gt_index[expected.correct_gt_ids[0]]
    return classify_from_icd10(entry.icd_10)


@dataclass
class FindingsBucket:
    """TP/TN/FP/FN tratando 'sospechoso' como 'lesión detectada' (positivo)."""

    true_positive: bool = False
    true_negative: bool = False
    false_positive: bool = False
    false_negative: bool = False


def findings_confusion_bucket(
    output: Component2Output, expected_class: Classification
) -> FindingsBucket:
    predicted_positive = output.classification == "sospechoso"
    actually_positive = expected_class == "sospechoso"

    if actually_positive and predicted_positive:
        return FindingsBucket(true_positive=True)
    if actually_positive and not predicted_positive:
        return FindingsBucket(false_negative=True)
    if not actually_positive and predicted_positive:
        return FindingsBucket(false_positive=True)
    return FindingsBucket(true_negative=True)


def _tokenize_zone_text(text: str) -> set[str]:
    return {w for w in text.lower().replace(",", " ").split() if len(w) > 2}


def segmentation_zone_iou_proxy(output: Component2Output, correct_entry: GroundTruthEntry) -> float:
    """
    Proxy de IoU: en vez de superposición de píxeles (no hay máscaras
    reales), calcula superposición de PALABRAS (Jaccard) entre las zonas
    que reportó C2 y las zonas/landmarks reales de la condición correcta.
    """
    reported_text = " ".join(roi.location for roi in output.segmentation.regions_of_interest)
    reported_zones = _tokenize_zone_text(reported_text)

    guidance = correct_entry.radiologist_guidance.imaging_location
    real_text = guidance.anatomical_landmarks + " " + " ".join(guidance.priority_zones)
    real_zones = _tokenize_zone_text(real_text)

    if not reported_zones or not real_zones:
        return 0.0

    intersection = len(reported_zones & real_zones)
    union = len(reported_zones | real_zones)
    return intersection / union if union > 0 else 0.0


@dataclass
class Component2MetricsReport:
    n_cases: int
    n_casos_clasificacion_aplicable: int
    sensibilidad_hallazgos: float
    especificidad_hallazgos: float
    segmentacion_iou_proxy_promedio: float
    concordancia_clinica: float | None
    detalle_por_caso: list[dict] = field(default_factory=list)


def aggregate_component2_metrics(
    cases: list[tuple[Component2Output, ExpectedOutput]],
    gt_index: dict[str, GroundTruthEntry],
) -> Component2MetricsReport:
    """
    Agrega las métricas de la tabla de C2 sobre una lista de casos que
    efectivamente llegaron a Componente 2 (ya filtrados por DERIVAR_A_IMAGEN
    en Paso 16) -- pensado para que el runner batch le pase los ~78 casos
    que activan C2, junto con el índice completo de la base GT.
    """
    n_cases = len(cases)
    if n_cases == 0:
        raise ValueError("Se necesita al menos un caso para calcular métricas")

    buckets: list[FindingsBucket] = []
    iou_scores: list[float] = []
    applicable = 0
    confidences: list[float] = []
    correctness: list[int] = []
    detalle = []

    for output, expected in cases:
        exp_class = expected_classification(expected, gt_index)
        is_applicable = exp_class is not None

        row = {"case_id": expected.case_id, "classification_applicable": is_applicable}

        if is_applicable:
            applicable += 1
            bucket = findings_confusion_bucket(output, exp_class)
            buckets.append(bucket)

            correct_entry = gt_index[expected.correct_gt_ids[0]]
            iou = segmentation_zone_iou_proxy(output, correct_entry)
            iou_scores.append(iou)

            is_correct = output.classification == exp_class
            confidences.append(output.confidence)
            correctness.append(int(is_correct))

            row.update(
                {
                    "expected_classification": exp_class,
                    "output_classification": output.classification,
                    "classification_correct": is_correct,
                    "segmentation_iou_proxy": iou,
                }
            )

        detalle.append(row)

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

    concordancia = None
    if len(confidences) >= 2 and len(set(correctness)) > 1:
        concordancia = float(np.corrcoef(confidences, correctness)[0, 1])

    return Component2MetricsReport(
        n_cases=n_cases,
        n_casos_clasificacion_aplicable=applicable,
        sensibilidad_hallazgos=sensibilidad,
        especificidad_hallazgos=especificidad,
        segmentacion_iou_proxy_promedio=(
            sum(iou_scores) / len(iou_scores) if iou_scores else float("nan")
        ),
        concordancia_clinica=concordancia,
        detalle_por_caso=detalle,
    )
