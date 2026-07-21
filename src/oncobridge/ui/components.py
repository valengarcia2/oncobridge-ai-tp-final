"""Componentes de UI reutilizables entre la vista de oncólogo y la de radiólogo."""

import html as _html

import streamlit as st

from oncobridge.schemas.component1_io import MatchedHypothesis
from oncobridge.schemas.component2_io import RegionOfInterest

# --- Traducciones a lenguaje natural de los valores internos del sistema
# (nunca se le muestra al usuario un literal tipo "DERIVAR_A_IMAGEN") ---
URGENCY_LABELS = {"alta": "Urgencia alta", "media": "Urgencia media", "baja": "Urgencia baja", "ninguna": "Sin urgencia"}
URGENCY_TONE = {"alta": "danger", "media": "warning", "baja": "success", "ninguna": "neutral"}

CLASSIFICATION_LABELS = {
    "sospechoso": "Hallazgo sospechoso",
    "benigno": "Hallazgo benigno",
    "no_concluyente": "No concluyente",
}
CLASSIFICATION_TONE = {"sospechoso": "danger", "benigno": "success", "no_concluyente": "warning"}

RECOMMENDATION_LABELS = {
    "DERIVAR_A_IMAGEN": "Derivar a estudio de imagen",
    "NO_DERIVAR": "No es necesario derivar a imagen",
    "SEGUIMIENTO_CLINICO": "Seguimiento clínico",
    "SIN_ELEMENTOS_PARA_EVALUAR": "Sin elementos suficientes para evaluar",
}
RECOMMENDATION_TONE = {
    "DERIVAR_A_IMAGEN": "danger",
    "NO_DERIVAR": "success",
    "SEGUIMIENTO_CLINICO": "warning",
    "SIN_ELEMENTOS_PARA_EVALUAR": "neutral",
}


def esc(text) -> str:
    """Escapa texto libre (LLM/GT) antes de insertarlo en un bloque HTML propio."""
    return _html.escape(str(text))


def _format_modality(raw: str) -> str:
    """'abdominal_MRI' -> 'Abdominal MRI' -- sin guiones bajos, con mayúscula donde corresponde."""
    words = raw.replace("_", " ").split()
    return " ".join(w if w.isupper() else w.capitalize() for w in words)


def inject_custom_css() -> None:
    """
    Hoja de estilos compartida por toda la app -- tarjetas blancas con
    sombra suave sobre el fondo celeste institucional, badges, barra de
    progreso propia, botones y el expander de "instrucciones" reestilizado
    como botón. Se llama una sola vez, al arrancar app.py. Usa
    unsafe_allow_html porque Streamlit no expone estos elementos como
    widgets nativos con este nivel de control visual.
    """
    st.markdown(
        """
        <style>
        /* --- Tipografía general --- */
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stCaptionContainer"] {
            font-size: 1.05rem;
            line-height: 1.6;
        }
        h1 { font-size: 2.2rem !important; }
        h2 { font-size: 1.65rem !important; margin-top: 0.4rem !important; }
        h3 { font-size: 1.3rem !important; margin-top: 0.6rem !important; }

        /* --- Tarjetas base: blancas, borde suave, sombra ligera --- */
        .ob-card, .ob-hypothesis-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 14px;
            padding: 18px 22px;
            margin-bottom: 18px;
            box-shadow: 0 2px 8px rgba(30, 64, 175, 0.06);
        }
        .ob-hypothesis-card { border-left: 5px solid #2563EB; margin-bottom: 22px; }
        .ob-card-title {
            font-size: 1.2rem;
            font-weight: 700;
            color: #0F172A;
            margin-bottom: 10px;
        }
        .ob-card-body p { margin: 0 0 8px 0; font-size: 1.05rem; color: #334155; }
        .ob-card-body ul { margin: 4px 0 0 0; padding-left: 20px; }
        .ob-card-body li { margin-bottom: 4px; font-size: 1.05rem; color: #334155; }

        /* --- Badges --- */
        .ob-badge {
            display: inline-block;
            padding: 8px 20px;
            border-radius: 999px;
            font-weight: 700;
            font-size: 1rem;
            letter-spacing: 0.2px;
            margin: 4px 8px 10px 0;
        }
        .ob-badge-danger { background: #FDE2E1; color: #9F1C13; }
        .ob-badge-warning { background: #FDECC8; color: #92400E; }
        .ob-badge-success { background: #D9F2E3; color: #05603A; }
        .ob-badge-neutral { background: #DCEAFB; color: #1D4ED8; }

        /* --- Tarjeta de confianza / probabilidad --- */
        .ob-confidence-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 14px;
            padding: 18px 22px;
            margin-bottom: 18px;
            box-shadow: 0 2px 8px rgba(30, 64, 175, 0.06);
        }
        .ob-confidence-label { font-size: 1rem; color: #475569; margin-bottom: 6px; font-weight: 600; }
        .ob-confidence-value { font-size: 2.4rem; font-weight: 800; color: #1D4ED8; line-height: 1.1; }
        .ob-progress-track {
            background: #E2ECFA;
            border-radius: 999px;
            height: 12px;
            margin-top: 12px;
            overflow: hidden;
        }
        .ob-progress-fill {
            background: linear-gradient(90deg, #3B82F6, #1D4ED8);
            height: 100%;
            border-radius: 999px;
        }

        /* --- Regiones de interés --- */
        .ob-roi-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 14px 18px;
            margin-bottom: 14px;
            box-shadow: 0 2px 8px rgba(30, 64, 175, 0.05);
        }
        .ob-roi-title { font-weight: 700; color: #0F172A; margin-bottom: 6px; font-size: 1.05rem; }
        .ob-roi-id { color: #64748B; font-weight: 400; font-size: 0.9rem; }
        .ob-roi-list { margin: 4px 0 0 0; padding-left: 20px; color: #334155; }
        .ob-roi-list li { margin-bottom: 4px; font-size: 1rem; }

        /* --- Botones: azules, texto blanco, hover visible --- */
        .stButton > button {
            border-radius: 8px;
            font-weight: 600;
            font-size: 1rem;
            padding: 0.5rem 1.2rem;
            transition: filter 0.15s ease, transform 0.05s ease;
        }
        .stButton > button:hover { filter: brightness(0.93); transform: translateY(-1px); }

        /* --- Expander "Instrucciones para el especialista" como botón azul --- */
        [data-testid="stExpander"] { border: none !important; box-shadow: none !important; }
        [data-testid="stExpander"] summary {
            background-color: #2563EB !important;
            color: #FFFFFF !important;
            border-radius: 8px !important;
            padding: 10px 18px !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
            list-style: none;
        }
        [data-testid="stExpander"] summary:hover { background-color: #1D4ED8 !important; }
        [data-testid="stExpander"] summary svg { fill: #FFFFFF !important; }
        [data-testid="stExpander"] summary::-webkit-details-marker { display: none; }
        [data-testid="stExpander"] > div:last-child {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 0 0 10px 10px;
            padding: 16px 20px;
            margin-top: 2px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_disclaimer() -> None:
    st.info(
        "**Este sistema asiste, no reemplaza.** Las recomendaciones son apoyo a la "
        "decisión clínica — la decisión diagnóstica y terapéutica final es siempre "
        "del médico especialista.",
        icon="⚕️",
    )


def render_step_header(current: str) -> None:
    """
    Header tipo wizard (1. Consulta Oncológica -> 2. Informe de Imágenes).
    Deja explícito que es un flujo secuencial de UN caso, no dos vistas
    independientes -- aunque el mecanismo por debajo (session_state.view)
    haga que se sienta como cambiar de pestaña.
    """
    paso1 = "**1. Consulta Oncológica**" if current == "oncologo" else "1. Consulta Oncológica"
    paso2 = "**2. Informe de Imágenes**" if current == "radiologo" else "2. Informe de Imágenes"
    paso3 = "**3. Encuesta**" if current == "encuesta" else "3. Encuesta"
    st.markdown(f"{paso1} &nbsp;→&nbsp; {paso2} &nbsp;→&nbsp; {paso3}")
    st.divider()


def render_info_card(title: str, html_body: str) -> None:
    """Tarjeta blanca genérica con título destacado y cuerpo en HTML ya armado."""
    st.markdown(
        f"""
        <div class="ob-card">
            <div class="ob-card-title">{title}</div>
            <div class="ob-card-body">{html_body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_badge(label: str, tone: str) -> None:
    st.markdown(f'<span class="ob-badge ob-badge-{tone}">{label}</span>', unsafe_allow_html=True)


def render_urgency_badge(urgency: str) -> None:
    _render_badge(URGENCY_LABELS.get(urgency, urgency), URGENCY_TONE.get(urgency, "neutral"))


def render_classification_badge(classification: str) -> None:
    _render_badge(CLASSIFICATION_LABELS.get(classification, classification), CLASSIFICATION_TONE.get(classification, "neutral"))


def render_recommendation_badge(recommendation: str) -> None:
    _render_badge(RECOMMENDATION_LABELS.get(recommendation, recommendation), RECOMMENDATION_TONE.get(recommendation, "neutral"))


def render_confidence_card(label: str, value: float) -> None:
    st.markdown(
        f"""
        <div class="ob-confidence-card">
            <div class="ob-confidence-label">{esc(label)}</div>
            <div class="ob-confidence-value">{value:.0%}</div>
            <div class="ob-progress-track">
                <div class="ob-progress-fill" style="width:{value * 100:.0f}%"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hypothesis_card(hypothesis: MatchedHypothesis, show_radiologist_instructions: bool = True) -> None:
    st.markdown(
        f"""
        <div class="ob-hypothesis-card">
            <strong>{esc(hypothesis.icd_10_description)}</strong> &nbsp;·&nbsp; {esc(hypothesis.icd_10)}
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_confidence_card("Probabilidad de match", hypothesis.match_probability)
    st.caption(hypothesis.match_rationale)
    if show_radiologist_instructions:
        with st.expander("Instrucciones para el especialista en imágenes"):
            ri = hypothesis.radiologist_instructions
            modalidades = ", ".join(_format_modality(m) for m in ri.suggested_modalities)
            st.markdown(f"**Modalidades sugeridas:** {modalidades}")
            st.markdown(f"**Zona anatómica:** {ri.imaging_location.body_region}")
    st.markdown("<div style='margin-bottom:6px'></div>", unsafe_allow_html=True)


def render_regions_of_interest(regions: list[RegionOfInterest]) -> None:
    """Detalle de las zonas que Componente 2 marcó como regiones de interés."""
    if not regions:
        st.caption("Sin regiones de interés reportadas.")
        return
    for roi in regions:
        st.markdown(
            f"""
            <div class="ob-roi-card">
                <div class="ob-roi-title">{esc(roi.location)} <span class="ob-roi-id">{esc(roi.id)}</span></div>
                <ul class="ob-roi-list">
                    <li>Tamaño estimado: {roi.size_mm:.1f} mm</li>
                    <li>Forma: {esc(roi.shape)}</li>
                    <li>Márgenes: {esc(roi.margins)}</li>
                    <li>Densidad: {esc(roi.density)}</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_error(exc: Exception) -> None:
    """
    Mensaje de error uniforme para cuando falla la llamada al LLM (cuota
    agotada, modelo no disponible, etc.) -- evita mostrarle al usuario un
    traceback crudo de Streamlit en medio de una demo.
    """
    st.error(
        "No se pudo completar el análisis. Puede ser un problema temporal de la "
        "API del modelo (cuota, límite de solicitudes, disponibilidad) -- probá "
        "de nuevo en unos segundos.\n\n"
        f"Detalle técnico: {exc}"
    )
