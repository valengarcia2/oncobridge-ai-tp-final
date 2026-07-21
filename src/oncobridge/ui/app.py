"""
OncoBridge AI — punto de entrada único de la interfaz Streamlit.

Flujo secuencial de un solo caso, no pestañas independientes elegibles:
el oncólogo analiza al paciente (Componente 1) y, si corresponde, deriva
explícitamente al especialista en imágenes (Componente 2) con un botón --
nunca automático, para reforzar "el sistema asiste, no reemplaza" y para
no gastar una llamada al LLM en casos donde el médico decide no derivar
aunque el sistema lo sugiera.

st.session_state.view controla qué vista se renderiza ("oncologo" o
"radiologo"). Streamlit no permite cambiar de pestaña nativa (st.tabs)
por código -- solo reacciona al click del usuario -- así que el cambio de
"pestaña" se simula con este estado propio en vez del widget nativo.

El disclaimer y el header de pasos quedan arriba de TODO, visibles sin
importar en qué vista/paso esté el usuario.
"""

import streamlit as st
import streamlit.components.v1 as components

from oncobridge.ui.components import inject_custom_css, render_disclaimer, render_step_header
from oncobridge.ui.views.encuesta import render_encuesta_view
from oncobridge.ui.views.oncologo import render_oncologo_view
from oncobridge.ui.views.radiologo import render_radiologo_view

st.set_page_config(page_title="OncoBridge AI", page_icon="🩺", layout="wide")
inject_custom_css()

if "view" not in st.session_state:
    st.session_state.view = "oncologo"

# Streamlit no vuelve solo al principio de la página al cambiar de vista
# (session_state.view) -- sin esto, el usuario queda con el scroll donde
# estaba en la vista anterior y hay que scrollear manualmente para leer
# desde arriba.
if st.session_state.get("_last_view") != st.session_state.view:
    st.session_state["_last_view"] = st.session_state.view
    components.html(
        """
        <script>
            var doc = window.parent.document;
            var container = doc.querySelector('section.main') || doc.querySelector('[data-testid="stAppViewContainer"]');
            if (container) { container.scrollTo({top: 0, behavior: 'instant'}); }
            window.parent.scrollTo(0, 0);
        </script>
        """,
        height=0,
    )

st.title("🩺 OncoBridge AI")
render_disclaimer()
render_step_header(current=st.session_state.view)

if st.session_state.view == "oncologo":
    render_oncologo_view()
elif st.session_state.view == "radiologo":
    render_radiologo_view()
else:
    render_encuesta_view()
