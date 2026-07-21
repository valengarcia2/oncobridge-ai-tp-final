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

from oncobridge.ui.components import inject_custom_css, render_disclaimer, render_step_header
from oncobridge.ui.views.oncologo import render_oncologo_view
from oncobridge.ui.views.radiologo import render_radiologo_view

st.set_page_config(page_title="OncoBridge AI", page_icon="🩺", layout="wide")
inject_custom_css()

if "view" not in st.session_state:
    st.session_state.view = "oncologo"

st.title("🩺 OncoBridge AI")
render_disclaimer()
render_step_header(current=st.session_state.view)

if st.session_state.view == "oncologo":
    render_oncologo_view()
else:
    render_radiologo_view()
