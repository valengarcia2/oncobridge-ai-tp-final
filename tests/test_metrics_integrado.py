"""
Tests de las métricas de Sistema Integrado. Todo sintético y sin LLM --
corre siempre, instantáneo.
"""

import time

import pytest

from oncobridge.evaluation.metrics_integrado import (
    compute_triage_time_reduction,
    tasa_imagen_innecesaria,
    time_call,
)
from oncobridge.schemas.component1_io import Component1Output, TokenUsage
from oncobridge.schemas.evaluation import ExpectedOutput


def _token_usage() -> TokenUsage:
    return TokenUsage(prompt_tokens=100, completion_tokens=100, total_tokens=200, model="test")


def _c1_output(recommendation: str) -> Component1Output:
    return Component1Output(
        patient_id="PAT-TEST",
        clinical_summary="test",
        matched_ground_truths=[],
        imaging_needed_probability=0.5,
        reasoning="test",
        recommendation=recommendation,
        urgency="alta" if recommendation == "DERIVAR_A_IMAGEN" else "ninguna",
        conclusive=True,
        token_usage=_token_usage(),
    )


def _expected(case_id: str, imaging_needed_ground_truth: bool) -> ExpectedOutput:
    return ExpectedOutput(
        case_id=case_id,
        correct_gt_ids=[],
        acceptable_secondary_gt_ids=[],
        imaging_needed_ground_truth=imaging_needed_ground_truth,
        urgency_ground_truth="alta" if imaging_needed_ground_truth else "ninguna",
        specialist_decision="DERIVAR_A_IMAGEN" if imaging_needed_ground_truth else "NO_DERIVAR",
        conclusive_ground_truth=True,
        difficulty="facil",
    )


def test_tasa_imagen_innecesaria_all_true_positives_is_zero():
    cases = [
        (_c1_output("DERIVAR_A_IMAGEN"), _expected("case_1", imaging_needed_ground_truth=True)),
        (_c1_output("DERIVAR_A_IMAGEN"), _expected("case_2", imaging_needed_ground_truth=True)),
    ]
    assert tasa_imagen_innecesaria(cases) == 0.0


def test_tasa_imagen_innecesaria_all_false_positives_is_one():
    cases = [
        (_c1_output("DERIVAR_A_IMAGEN"), _expected("case_1", imaging_needed_ground_truth=False)),
        (_c1_output("DERIVAR_A_IMAGEN"), _expected("case_2", imaging_needed_ground_truth=False)),
    ]
    assert tasa_imagen_innecesaria(cases) == 1.0


def test_tasa_imagen_innecesaria_mixed_cases():
    cases = [
        (_c1_output("DERIVAR_A_IMAGEN"), _expected("case_1", imaging_needed_ground_truth=True)),  # TP
        (_c1_output("DERIVAR_A_IMAGEN"), _expected("case_2", imaging_needed_ground_truth=False)),  # FP
        (_c1_output("DERIVAR_A_IMAGEN"), _expected("case_3", imaging_needed_ground_truth=True)),  # TP
        (_c1_output("DERIVAR_A_IMAGEN"), _expected("case_4", imaging_needed_ground_truth=False)),  # FP
    ]
    # 2 TP, 2 FP -> 2 / (2+2) = 0.5
    assert tasa_imagen_innecesaria(cases) == 0.5


def test_tasa_imagen_innecesaria_ignores_cases_where_no_image_was_requested():
    """TN/FN no piden imagen -- no forman parte del denominador de esta métrica."""
    cases = [
        (_c1_output("DERIVAR_A_IMAGEN"), _expected("case_1", imaging_needed_ground_truth=True)),  # TP
        (_c1_output("NO_DERIVAR"), _expected("case_2", imaging_needed_ground_truth=False)),  # TN
        (_c1_output("NO_DERIVAR"), _expected("case_3", imaging_needed_ground_truth=True)),  # FN
    ]
    assert tasa_imagen_innecesaria(cases) == 0.0


def test_tasa_imagen_innecesaria_requires_at_least_one_case():
    with pytest.raises(ValueError):
        tasa_imagen_innecesaria([])


def test_time_call_returns_result_and_positive_elapsed_seconds():
    def _slow_add(a, b):
        time.sleep(0.01)
        return a + b

    result, elapsed = time_call(_slow_add, 2, 3)
    assert result == 5
    assert elapsed > 0


def test_compute_triage_time_reduction_uses_measured_average():
    report = compute_triage_time_reduction(
        measured_seconds=[4.0, 6.0],
        referencia_manual_minutos=5.0,
        referencia_fuente="fuente de prueba",
    )
    assert report.measured_seconds_promedio == pytest.approx(5.0)
    assert report.referencia_manual_minutos == 5.0
    assert report.referencia_fuente == "fuente de prueba"
    # 5 seg = 0.0833 min vs referencia de 5 min -> reduccion cercana al 100%
    assert report.reduccion_pct == pytest.approx(98.33, abs=0.1)


def test_compute_triage_time_reduction_has_real_default_citation():
    report = compute_triage_time_reduction(measured_seconds=[3.0])
    assert "Overhage" in report.referencia_fuente
    assert "Annals of Internal Medicine" in report.referencia_fuente
    assert report.referencia_manual_minutos == pytest.approx(5.36, abs=0.01)


def test_compute_triage_time_reduction_requires_measurements():
    with pytest.raises(ValueError):
        compute_triage_time_reduction(measured_seconds=[])


def test_compute_triage_time_reduction_requires_a_cited_source():
    with pytest.raises(ValueError):
        compute_triage_time_reduction(measured_seconds=[3.0], referencia_fuente="")
