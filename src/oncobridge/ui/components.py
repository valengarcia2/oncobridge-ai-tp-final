"""Componentes de UI reutilizables entre la vista de oncólogo y la de radiólogo."""

import streamlit as st

from oncobridge.schemas.component1_io import MatchedHypothesis

URGENCY_ICON = {"alta": "🔴", "media": "🟠", "baja": "🟢", "ninguna": "⚪"}


def render_disclaimer() -> None:
    st.info(
        "**Este sistema asiste, no reemplaza.** Las recomendaciones son apoyo a la "
        "decisión clínica — la decisión diagnóstica y terapéutica final es siempre "
        "del médico especialista.",
        icon="⚕️",
    )


def render_urgency_badge(urgency: str) -> str:
    return f"{URGENCY_ICON.get(urgency, '⚪')} Urgencia: **{urgency.upper()}**"


def render_hypothesis_card(hypothesis: MatchedHypothesis) -> None:
    with st.container(border=True):
        st.markdown(f"**{hypothesis.icd_10_description}** · `{hypothesis.icd_10}`")
        st.progress(
            hypothesis.match_probability,
            text=f"Probabilidad de match: {hypothesis.match_probability:.0%}",
        )
        st.caption(hypothesis.match_rationale)
        with st.expander("Instrucciones para el especialista en imágenes"):
            ri = hypothesis.radiologist_instructions
            st.markdown(f"**Modalidades sugeridas:** {', '.join(ri.suggested_modalities)}")
            st.markdown(f"**Zona anatómica:** {ri.imaging_location.body_region}")
            st.markdown(f"**Contexto clínico:** {ri.clinical_context_for_radiologist}")
