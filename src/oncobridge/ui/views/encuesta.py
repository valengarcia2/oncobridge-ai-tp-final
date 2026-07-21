"""
Vista de encuesta de evaluación por especialista. Guarda con la MISMA
función (save_response) que usa el script de consola
scripts/collect_specialist_feedback.py -- una respuesta cargada acá queda
en evaluation/results/specialist_feedback.json y se promedia
automáticamente en el próximo reporte de evaluación, sin ningún paso de
integración adicional.

El texto de esta vista está pensado para quien completa la encuesta (un
especialista), no para el equipo -- evitar jerga interna del proyecto.
"""

import streamlit as st

from oncobridge.evaluation.specialist_feedback import save_response


def render_encuesta_view() -> None:
    st.header("Encuesta de evaluación")
    st.write("Tu opinión nos ayuda a mejorar el sistema. Contanos cómo fue tu experiencia.")

    with st.form("encuesta_form"):
        nps_score = st.slider(
            "¿Qué tan probable es que recomendarías este sistema a un colega?",
            min_value=1, max_value=5, value=4,
        )
        coherencia_razonamiento = st.slider(
            "Coherencia del razonamiento clínico (Consulta Oncológica)",
            min_value=1, max_value=5, value=4,
        )
        completitud_informe = st.slider(
            "Completitud del informe de imágenes", min_value=1, max_value=5, value=4
        )
        precision_informe = st.slider(
            "Precisión del informe de imágenes", min_value=1, max_value=5, value=4
        )
        claridad_informe = st.slider(
            "Claridad del informe de imágenes", min_value=1, max_value=5, value=4
        )
        comentario = st.text_area("Comentario libre (opcional)")

        enviado = st.form_submit_button("Enviar encuesta", type="primary")

    if enviado:
        save_response(
            {
                "nps_score": nps_score,
                "coherencia_razonamiento": coherencia_razonamiento,
                "completitud_informe": completitud_informe,
                "precision_informe": precision_informe,
                "claridad_informe": claridad_informe,
                "comentario": comentario,
            }
        )
        st.success("¡Gracias! Tu respuesta se guardó correctamente.")

    st.divider()
    if st.button("← Analizar otro paciente", type="primary"):
        for key in ("c1_output", "c2_output", "patient"):
            st.session_state.pop(key, None)
        st.session_state.view = "oncologo"
        st.rerun()
