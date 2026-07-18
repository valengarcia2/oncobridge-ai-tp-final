"""
Tests del motor de probabilidad. Lógica pura (sin LLM), corren sin
necesitar API key ni internet.
"""

import pytest

from oncobridge.component1.probability_engine import (
    compute_imaging_needed_probability,
    HypothesisForProbability,
)


def test_no_hypotheses_not_conclusive_returns_sin_elementos():
    prob, recommendation, urgency = compute_imaging_needed_probability([], conclusive=False)
    assert prob == 0.0
    assert recommendation == "SIN_ELEMENTOS_PARA_EVALUAR"
    assert urgency == "ninguna"


def test_no_hypotheses_but_conclusive_returns_no_derivar():
    prob, recommendation, urgency = compute_imaging_needed_probability([], conclusive=True)
    assert prob == 0.0
    assert recommendation == "NO_DERIVAR"
    assert urgency == "ninguna"


def test_single_high_probability_high_urgency_hypothesis_derives():
    hyps = [HypothesisForProbability(match_probability=0.9, urgency_level="alta")]
    prob, recommendation, urgency = compute_imaging_needed_probability(hyps, conclusive=True)
    assert prob == pytest.approx(0.9, abs=0.01)
    assert recommendation == "DERIVAR_A_IMAGEN"
    assert urgency == "alta"


def test_takes_max_not_average_across_hypotheses():
    hyps = [
        HypothesisForProbability(match_probability=0.85, urgency_level="alta"),
        HypothesisForProbability(match_probability=0.2, urgency_level="baja"),
    ]
    prob, recommendation, urgency = compute_imaging_needed_probability(hyps, conclusive=True)
    assert prob == pytest.approx(0.85, abs=0.01)
    assert recommendation == "DERIVAR_A_IMAGEN"
    assert urgency == "alta"


def test_low_probability_low_urgency_does_not_derive():
    hyps = [HypothesisForProbability(match_probability=0.3, urgency_level="baja")]
    prob, recommendation, urgency = compute_imaging_needed_probability(hyps, conclusive=True)
    assert prob == pytest.approx(0.15, abs=0.01)
    assert recommendation == "NO_DERIVAR"
    assert urgency == "ninguna"


def test_intermediate_probability_triggers_seguimiento_clinico():
    hyps = [HypothesisForProbability(match_probability=0.5, urgency_level="media")]
    prob, recommendation, urgency = compute_imaging_needed_probability(hyps, conclusive=True)
    assert recommendation == "SEGUIMIENTO_CLINICO"
    assert urgency == "baja"


def test_probability_is_bounded_between_0_and_1():
    hyps = [HypothesisForProbability(match_probability=1.0, urgency_level="alta")]
    prob, _, _ = compute_imaging_needed_probability(hyps, conclusive=True)
    assert 0.0 <= prob <= 1.0