"""
Formulario para cargar un paciente nuevo (no del dataset), siguiendo el
schema real de PatientInput -- lo que se completa acá es válido sin
transformación adicional.
"""

import uuid

import pandas as pd
import streamlit as st

from oncobridge import config
from oncobridge.schemas.component1_io import Demographics, MedicalHistoryEvent, PatientInput

_EMPTY_HISTORY_ROW = {"fecha": "", "evento": ""}
_EMPTY_LAB_ROW = {"nombre": "", "valor": ""}


def render_new_patient_form() -> PatientInput | None:
    """Devuelve un PatientInput si el usuario completó y confirmó el formulario; si no, None."""
    st.markdown("#### Datos del paciente")

    patient_id = st.text_input(
        "ID del paciente", value=f"PAT-{uuid.uuid4().hex[:6].upper()}"
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
    st.caption(
        f"Más de {config.COMPLEX_HISTORY_THRESHOLD} eventos activa el resumen "
        "automático antes de mandarlo al modelo."
    )
    history_df = st.data_editor(
        pd.DataFrame([_EMPTY_HISTORY_ROW]),
        num_rows="dynamic",
        width="stretch",
        key="medical_history_editor",
    )

    st.markdown("##### Laboratorio actual")
    labs_df = st.data_editor(
        pd.DataFrame([_EMPTY_LAB_ROW]),
        num_rows="dynamic",
        width="stretch",
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
