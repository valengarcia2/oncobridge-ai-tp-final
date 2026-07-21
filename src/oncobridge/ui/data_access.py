"""
Puente entre la UI de Streamlit y los pipelines REALES de C1/C2.

gt_entries y el retriever se cachean a nivel de recurso de Streamlit
(st.cache_resource): son costosos de construir (cargan el modelo de
embeddings) y no cambian entre pacientes, así que se arman una sola vez
por sesión del server, no en cada rerun del script.
"""

import json

import streamlit as st

from oncobridge import config
from oncobridge.component1.pipeline import run_component1
from oncobridge.component2.pipeline import run_component2
from oncobridge.ingestion.gt_loader import load_ground_truth_base
from oncobridge.rag.retriever import HybridRetriever
from oncobridge.schemas.component1_io import Component1Output, PatientInput
from oncobridge.schemas.component2_io import Component2Output


def list_available_cases() -> list[str]:
    return sorted(p.name for p in config.CLINICAL_CASES_DIR.glob("case_*"))


def load_patient_input(case_id: str) -> PatientInput:
    raw = json.loads((config.CLINICAL_CASES_DIR / case_id / "input.json").read_text(encoding="utf-8"))
    return PatientInput.model_validate(raw)


@st.cache_resource(show_spinner=False)
def _get_gt_entries_and_retriever():
    gt_entries = load_ground_truth_base(config.GT_BASE_DIR)
    retriever = HybridRetriever()
    retriever.build(gt_entries)
    return gt_entries, retriever


def analyze_patient(patient: PatientInput) -> Component1Output:
    """Corre Componente 1 de verdad -- llama al LLM configurado en .env."""
    gt_entries, retriever = _get_gt_entries_and_retriever()
    return run_component1(patient, gt_entries=gt_entries, retriever=retriever)


def analyze_imaging(c1_output: Component1Output) -> Component2Output:
    """Corre Componente 2 de verdad -- se invoca solo cuando el médico deriva explícitamente."""
    gt_entries, _ = _get_gt_entries_and_retriever()
    return run_component2(c1_output, gt_entries=gt_entries)
