"""
Tests de las métricas de Componente 1. Todo sintético y sin LLM -- corre
siempre, instantáneo.
"""

import json

from oncobridge import config
from oncobridge.evaluation.metrics_component1 import (
    aggregate_component1_metrics,
    conclusive_matches_expected,
    gt_match_is_applicable,
    gt_match_is_correct,
    imaging_confusion_bucket,
    imaging_decision_is_correct,
)
from oncobridge.schemas.component1_io import (
    Component1Output,
    ImagingLocation,
    MatchedHypothesis,
    RadiologistInstructions,
    TokenUsage,
)
from oncobridge.schemas.evaluation import ExpectedOutput


def _imaging_location() -> ImagingLocation:
    return ImagingLocation(
        body_region="torax",
        anatomical_landmarks="lobulo superior",
        bilateral_comparison_required=False,
        priority_zones=["nodulo"],
        positioning_notes="",
    )


def _radiologist_instructions() -> RadiologistInstructions:
    return RadiologistInstructions(
        suggested_modalities=["chest_CT"],
        imaging_location=_imaging_location(),
        clinical_context_for_radiologist="",
        meddiffusion_reference_prompt="",
        meddiffusion_negative_prompt="",
        reference_images_note="",
    )


def _token_usage(total: int = 1000) -> TokenUsage:
    return TokenUsage(
        prompt_tokens=total - 200, completion_tokens=200, total_tokens=total, model="test-model"
    )


def _hypothesis(gt_id: str, match_probability: float) -> MatchedHypothesis:
    return MatchedHypothesis(
        gt_id=gt_id,
        icd_10="C00.0",
        icd_10_description="test",
        match_probability=match_probability,
        match_rationale="test",
        radiologist_instructions=_radiologist_instructions(),
    )


def _c1_output(
    matched=None,
    recommendation="DERIVAR_A_IMAGEN",
    urgency="alta",
    conclusive=True,
    imaging_needed_probability=0.9,
    total_tokens=1000,
) -> Component1Output:
    return Component1Output(
        patient_id="PAT-TEST",
        clinical_summary="test",
        matched_ground_truths=matched or [],
        imaging_needed_probability=imaging_needed_probability,
        reasoning="test",
        recommendation=recommendation,
        urgency=urgency,
        conclusive=conclusive,
        token_usage=_token_usage(total_tokens),
    )


def _expected(
    case_id="case_test",
    correct_gt_ids=None,
    acceptable_secondary_gt_ids=None,
    imaging_needed_ground_truth=True,
    urgency_ground_truth="alta",
    specialist_decision="DERIVAR_A_IMAGEN",
    conclusive_ground_truth=True,
    difficulty="facil",
) -> ExpectedOutput:
    return ExpectedOutput(
        case_id=case_id,
        correct_gt_ids=correct_gt_ids or [],
        acceptable_secondary_gt_ids=acceptable_secondary_gt_ids or [],
        imaging_needed_ground_truth=imaging_needed_ground_truth,
        urgency_ground_truth=urgency_ground_truth,
        specialist_decision=specialist_decision,
        conclusive_ground_truth=conclusive_ground_truth,
        difficulty=difficulty,
    )


def test_gt_match_is_correct_when_top_hypothesis_matches():
    output = _c1_output(matched=[_hypothesis("GT-A", 0.9), _hypothesis("GT-B", 0.3)])
    expected = _expected(correct_gt_ids=["GT-A"])
    assert gt_match_is_applicable(expected) is True
    assert gt_match_is_correct(output, expected) is True


def test_gt_match_is_incorrect_when_top_hypothesis_is_wrong():
    output = _c1_output(matched=[_hypothesis("GT-B", 0.9)])
    expected = _expected(correct_gt_ids=["GT-A"])
    assert gt_match_is_correct(output, expected) is False


def test_gt_match_not_applicable_without_correct_gt_ids():
    expected = _expected(correct_gt_ids=[])
    assert gt_match_is_applicable(expected) is False


def test_imaging_decision_correctness():
    output = _c1_output(recommendation="NO_DERIVAR")
    assert imaging_decision_is_correct(output, _expected(specialist_decision="NO_DERIVAR")) is True
    assert imaging_decision_is_correct(output, _expected(specialist_decision="DERIVAR_A_IMAGEN")) is False


def test_conclusive_matches_expected():
    output = _c1_output(conclusive=True)
    assert conclusive_matches_expected(output, _expected(conclusive_ground_truth=True)) is True
    assert conclusive_matches_expected(output, _expected(conclusive_ground_truth=False)) is False


def test_confusion_bucket_true_positive():
    output = _c1_output(recommendation="DERIVAR_A_IMAGEN")
    bucket = imaging_confusion_bucket(output, _expected(imaging_needed_ground_truth=True))
    assert bucket.true_positive is True


def test_confusion_bucket_false_negative():
    output = _c1_output(recommendation="NO_DERIVAR")
    bucket = imaging_confusion_bucket(output, _expected(imaging_needed_ground_truth=True))
    assert bucket.false_negative is True


def test_confusion_bucket_false_positive():
    output = _c1_output(recommendation="DERIVAR_A_IMAGEN")
    bucket = imaging_confusion_bucket(output, _expected(imaging_needed_ground_truth=False))
    assert bucket.false_positive is True


def test_confusion_bucket_true_negative():
    output = _c1_output(recommendation="NO_DERIVAR")
    bucket = imaging_confusion_bucket(output, _expected(imaging_needed_ground_truth=False))
    assert bucket.true_negative is True


def test_aggregate_metrics_over_synthetic_cases():
    pairs = [
        (
            _c1_output(matched=[_hypothesis("GT-A", 0.9)], recommendation="DERIVAR_A_IMAGEN"),
            _expected(correct_gt_ids=["GT-A"], specialist_decision="DERIVAR_A_IMAGEN", imaging_needed_ground_truth=True),
        ),
        (
            _c1_output(matched=[], recommendation="NO_DERIVAR", conclusive=False),
            _expected(
                correct_gt_ids=[],
                specialist_decision="NO_DERIVAR",
                imaging_needed_ground_truth=False,
                conclusive_ground_truth=False,
            ),
        ),
        (
            # gt match no aplicable (sin correct_gt_ids) pero SI se evalua la decision
            _c1_output(matched=[], recommendation="DERIVAR_A_IMAGEN"),
            _expected(
                correct_gt_ids=[],
                specialist_decision="DERIVAR_A_IMAGEN",
                imaging_needed_ground_truth=True,
            ),
        ),
    ]

    report = aggregate_component1_metrics(pairs)

    assert report.n_cases == 3
    assert report.accuracy_derivacion == 1.0  # los 3 acertaron la decision
    assert report.sensibilidad == 1.0  # 2 TP, 0 FN
    assert report.especificidad == 1.0  # 1 TN, 0 FP
    assert report.n_casos_gt_match_aplicable == 1  # solo el primer caso tiene correct_gt_ids
    assert report.precision_gt_match == 1.0
    assert report.tokens_promedio_por_caso == 1000


def test_real_case_with_correct_gt_ids_is_applicable_for_gt_match():
    """case_001 real tiene correct_gt_ids -- debe ser aplicable."""
    raw = json.loads(
        (config.CLINICAL_CASES_DIR / "case_001" / "expected_output.json").read_text(encoding="utf-8")
    )
    expected = ExpectedOutput.model_validate(raw)
    assert gt_match_is_applicable(expected) is True


def test_real_case_109_without_correct_gt_ids_is_not_applicable_for_gt_match():
    """
    case_109 real: DERIVAR_A_IMAGEN sin correct_gt_ids (la base GT no cubre
    la condición) -- debe quedar excluido del denominador de GT match,
    pero sigue siendo un caso valido para accuracy de derivacion.
    """
    raw = json.loads(
        (config.CLINICAL_CASES_DIR / "case_109" / "expected_output.json").read_text(encoding="utf-8")
    )
    expected = ExpectedOutput.model_validate(raw)
    assert expected.specialist_decision == "DERIVAR_A_IMAGEN"
    assert gt_match_is_applicable(expected) is False
