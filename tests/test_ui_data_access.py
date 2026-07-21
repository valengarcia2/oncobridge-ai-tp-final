"""
Tests de data_access.py. list_available_cases/load_patient_input son
puros (sin LLM, corren siempre). analyze_patient/analyze_imaging llaman
al pipeline real -- requieren GEMINI_API_KEY, mismo patrón que el resto
de los tests de integración del proyecto.
"""

import os

import pytest

from oncobridge.ui.data_access import (
    analyze_imaging,
    analyze_patient,
    list_available_cases,
    load_patient_input,
)


def test_list_available_cases_returns_110_cases():
    cases = list_available_cases()
    assert len(cases) == 110
    assert cases[0] == "case_001"


def test_load_patient_input_matches_schema():
    patient = load_patient_input("case_001")
    assert patient.patient_id == "PAT-00101"


@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="Requiere GEMINI_API_KEY real")
def test_analyze_patient_returns_valid_c1_output():
    patient = load_patient_input("case_001")
    output = analyze_patient(patient)

    assert output.patient_id == patient.patient_id
    assert output.token_usage.total_tokens > 0


@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="Requiere GEMINI_API_KEY real")
def test_analyze_imaging_returns_valid_c2_output_when_derived():
    patient = load_patient_input("case_001")
    c1_output = analyze_patient(patient)
    assert c1_output.recommendation == "DERIVAR_A_IMAGEN"

    c2_output = analyze_imaging(c1_output)

    assert c2_output.patient_id == c1_output.patient_id
    assert c2_output.classification in ("sospechoso", "benigno", "no_concluyente")
