"""
Vista del especialista en imágenes: Componente 2. Se llega acá solo por
el botón explícito de derivación de la vista del oncólogo -- nunca
automático.
"""

import streamlit as st

from oncobridge.component2.image_synthesizer import reference_image_path
from oncobridge.ui.components import (
    esc,
    render_classification_badge,
    render_confidence_card,
    render_error,
    render_info_card,
    render_regions_of_interest,
)
from oncobridge.ui.data_access import analyze_imaging


def render_radiologo_view() -> None:
    if st.button("← Volver a la consulta oncológica", type="primary"):
        st.session_state.view = "oncologo"
        st.rerun()

    st.header("Informe de Imágenes")

    c1_output = st.session_state.get("c1_output")
    if c1_output is None:
        st.warning("Todavía no hay ningún caso derivado desde la Consulta Oncológica.")
        return

    st.caption(f"Paciente derivado desde Consulta Oncológica: {c1_output.patient_id}")
    top = c1_output.matched_ground_truths[0]

    if "c2_output" not in st.session_state:
        with st.spinner("Generando informe de imágenes..."):
            try:
                st.session_state.c2_output = analyze_imaging(c1_output)
            except Exception as exc:
                render_error(exc)
                return

    output = st.session_state.c2_output

    col_imagen, col_resultado = st.columns([1, 1.4])

    with col_imagen:
        st.markdown("### Imagen de referencia")
        image_path = reference_image_path(top.gt_id)
        if image_path.exists():
            st.image(
                str(image_path),
                caption="Imagen generada por IA -- no corresponde al estudio real del paciente.",
                width=360,
            )
        else:
            st.caption("No hay imagen de referencia cacheada para esta hipótesis todavía.")

    with col_resultado:
        render_info_card("Hallazgos esperados", f"<p>{esc(output.findings)}</p>")
        render_classification_badge(output.classification)
        render_confidence_card("Confianza", output.confidence)

        st.markdown("### Regiones de interés")
        render_regions_of_interest(output.segmentation.regions_of_interest)

    st.divider()
    next_steps_html = ""
    if output.next_steps:
        items = "".join(f"<li>{esc(step)}</li>" for step in output.next_steps)
        next_steps_html = f"<p><strong>Próximos pasos sugeridos:</strong></p><ul>{items}</ul>"
    render_info_card("Recomendación final", f"<p>{esc(output.final_recommendation)}</p>{next_steps_html}")

    col_reiniciar, col_encuesta = st.columns(2)
    with col_reiniciar:
        if st.button("Analizar otro paciente", type="primary"):
            for key in ("c1_output", "c2_output", "patient"):
                st.session_state.pop(key, None)
            st.session_state.view = "oncologo"
            st.rerun()
    with col_encuesta:
        if st.button("Responder encuesta de evaluación →", type="primary"):
            st.session_state.view = "encuesta"
            st.rerun()
