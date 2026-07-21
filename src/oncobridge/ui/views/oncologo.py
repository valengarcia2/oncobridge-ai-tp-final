"""
Vista del oncólogo: selección de paciente (del dataset o nuevo) +
Componente 1. Termina en un botón explícito de derivación -- el sistema
sugiere, el médico decide, y esa decisión es la que gasta (o no) una
llamada al LLM de Componente 2.
"""

import streamlit as st

from oncobridge.ui.components import (
    esc,
    render_confidence_card,
    render_error,
    render_hypothesis_card,
    render_info_card,
    render_recommendation_badge,
    render_urgency_badge,
)
from oncobridge.ui.data_access import analyze_patient, list_available_cases, load_patient_input
from oncobridge.ui.patient_form import render_new_patient_form


def render_oncologo_view() -> None:
    st.header("Consulta Oncológica")

    modo = st.radio(
        "Paciente", ["Elegir del dataset", "Cargar paciente nuevo"], horizontal=True
    )

    patient = None
    if modo == "Elegir del dataset":
        case_id = st.selectbox(
            "Caso", list_available_cases(), format_func=lambda c: c.replace("case_", "Caso ")
        )
        if case_id and st.button("Analizar paciente", type="primary", key="analizar_dataset"):
            patient = load_patient_input(case_id)
    else:
        patient = render_new_patient_form()

    if patient is not None:
        with st.spinner("Analizando historial clínico..."):
            try:
                output = analyze_patient(patient)
            except Exception as exc:
                render_error(exc)
                return
        st.session_state.patient = patient
        st.session_state.c1_output = output
        st.session_state.pop("c2_output", None)  # nuevo análisis invalida cualquier informe viejo

    if "c1_output" not in st.session_state:
        return

    patient = st.session_state.patient
    output = st.session_state.c1_output

    col_datos, col_resultado = st.columns([1, 2])

    with col_datos:
        symptoms_html = "".join(
            f"<li>{esc(s[:1].upper() + s[1:])}</li>" for s in patient.current_symptoms
        )
        patient_html = (
            f"<p><strong>ID:</strong> {esc(patient.patient_id)}</p>"
            f"<p><strong>Edad / Sexo:</strong> {patient.demographics.age} / {esc(patient.demographics.sex)}</p>"
            "<p><strong>Síntomas actuales:</strong></p>"
            f"<ul>{symptoms_html}</ul>"
        )
        render_info_card("Datos del paciente", patient_html)

    with col_resultado:
        render_info_card("Resumen clínico", f"<p>{esc(output.clinical_summary)}</p>")

        st.markdown("### Recomendación")
        render_urgency_badge(output.urgency)
        render_confidence_card("Probabilidad de necesitar imagen", output.imaging_needed_probability)
        st.write("**Decisión sugerida:**")
        render_recommendation_badge(output.recommendation)

        derivar = output.recommendation == "DERIVAR_A_IMAGEN"
        if output.conclusive and output.matched_ground_truths:
            st.markdown("### Hipótesis diagnósticas")
            for hypothesis in output.matched_ground_truths:
                render_hypothesis_card(hypothesis, show_radiologist_instructions=derivar)
        else:
            st.warning(
                "El sistema no encontró evidencia suficiente para sostener una "
                "hipótesis oncológica con los datos disponibles."
            )

    st.divider()
    if output.recommendation == "DERIVAR_A_IMAGEN" and output.matched_ground_truths:
        st.caption(
            "El sistema sugiere derivar a estudio de imagen. La decisión de derivar "
            "es siempre del médico."
        )
        if st.button("Derivar a especialista en imágenes →", type="primary"):
            st.session_state.view = "radiologo"
            st.rerun()
    else:
        if st.button("Responder encuesta de evaluación →", type="primary"):
            st.session_state.view = "encuesta"
            st.rerun()
