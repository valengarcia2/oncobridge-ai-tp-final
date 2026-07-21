"""
Formulario para cargar un paciente nuevo (no del dataset), siguiendo el
schema real de PatientInput -- lo que se completa acá es válido sin
transformación adicional.
"""

import uuid

import pandas as pd
import streamlit as st

from oncobridge.schemas.component1_io import Demographics, MedicalHistoryEvent, PatientInput

_EMPTY_HISTORY_ROW = {"fecha": "", "evento": ""}
_EMPTY_LAB_ROW = {"nombre": "", "valor": ""}


def render_new_patient_form() -> PatientInput | None:
    """Devuelve un PatientInput si el usuario completó y confirmó el formulario; si no, None."""
    st.markdown("#### Datos del paciente")

    # Se genera una sola vez por sesión (no en cada rerun del script) y con
    # key fija -- si no, Streamlit lo trata como un widget nuevo en cada
    # interacción y el ID "cambia solo" cada vez que tocás otro campo.
    if "new_patient_id" not in st.session_state:
        st.session_state.new_patient_id = f"PAT-{uuid.uuid4().hex[:6].upper()}"

    patient_id = st.text_input(
        "ID del paciente", value=st.session_state.new_patient_id, key="patient_id_input"
    )

    col_edad, col_sexo = st.columns(2)
    age = col_edad.number_input("Edad", min_value=0, max_value=120, value=50)
    sex = col_sexo.selectbox("Sexo", options=["F", "M"])

    family_history_text = st.text_area(
        "Antecedentes familiares (uno por línea)",
        placeholder="carcinoma_renal\nhipertension",
    )
    current_symptoms_text = st.text_area(
        "Síntomas actuales (uno por línea)",
        placeholder="hematuria macroscopica intermitente 3 semanas\ndolor lumbar izquierdo persistente",
    )

    st.markdown("##### Historial clínico (eventos)")
    with st.container(border=True):
        history_df = st.data_editor(
            pd.DataFrame([_EMPTY_HISTORY_ROW]),
            num_rows="dynamic",
            width="stretch",
            hide_index=True,
            column_config={
                "fecha": st.column_config.TextColumn("Fecha", width="medium"),
                "evento": st.column_config.TextColumn("Evento", width="large"),
            },
            key="medical_history_editor",
        )

    st.markdown("##### Laboratorio actual")
    with st.container(border=True):
        labs_df = st.data_editor(
            pd.DataFrame([_EMPTY_LAB_ROW]),
            num_rows="dynamic",
            width="stretch",
            hide_index=True,
            column_config={
                "nombre": st.column_config.TextColumn("Nombre", width="medium"),
                "valor": st.column_config.TextColumn("Valor", width="medium"),
            },
            key="labs_editor",
        )

    if not st.button("Analizar paciente", type="primary", key="analizar_paciente_nuevo"):
        return None

    current_symptoms = [line.strip() for line in current_symptoms_text.splitlines() if line.strip()]
    if not current_symptoms:
        st.error("Ingresá al menos un síntoma antes de analizar.")
        return None

    family_history = [line.strip() for line in family_history_text.splitlines() if line.strip()]

    medical_history = [
        MedicalHistoryEvent(date=str(row["fecha"]), event=str(row["evento"]))
        for row in history_df.to_dict("records")
        if row.get("fecha") and row.get("evento")
    ]

    current_labs = {
        str(row["nombre"]): row["valor"]
        for row in labs_df.to_dict("records")
        if row.get("nombre")
    }

    return PatientInput(
        patient_id=patient_id,
        demographics=Demographics(age=int(age), sex=sex, family_history=family_history),
        current_symptoms=current_symptoms,
        medical_history=medical_history,
        current_labs=current_labs,
    )
