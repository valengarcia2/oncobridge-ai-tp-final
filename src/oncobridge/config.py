"""
Configuración centralizada de OncoBridge AI.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # sube desde src/oncobridge/
DATASET_ROOT = PROJECT_ROOT / os.getenv("DATASET_ROOT", "data/dataset_clinical_only/dataset")
GT_BASE_DIR = DATASET_ROOT / "oncology_ground_truth_base"
CLINICAL_CASES_DIR = DATASET_ROOT / "clinical_cases"
INDEX_JSON_PATH = DATASET_ROOT / "index.json"

# --- Modelos por tarea ---
# gemini-flash-latest: alias de Google que siempre apunta al modelo flash
# vigente. Usamos el alias (no una versión fija tipo "gemini-2.5-flash")
# porque Google puede dejar de dar acceso a una versión puntual a API keys
# nuevas sin previo aviso -- nos pasó en julio 2026 con gemini-2.5-flash
# (404 "no longer available to new users").
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
LLM_MODEL_REASONING = os.getenv("LLM_MODEL_REASONING", "gemini-flash-latest")
LLM_MODEL_SUMMARIZATION = os.getenv("LLM_MODEL_SUMMARIZATION", "gemini-flash-latest")
LLM_MODEL_VISION = os.getenv("LLM_MODEL_VISION", "gemini-flash-latest")

# --- Umbrales de eficiencia de contexto ---
COMPLEX_HISTORY_THRESHOLD = int(os.getenv("COMPLEX_HISTORY_THRESHOLD", "5"))
RETRIEVER_TOP_K = int(os.getenv("RETRIEVER_TOP_K", "5"))
RETRIEVER_MIN_SCORE = float(os.getenv("RETRIEVER_MIN_SCORE", "0.15"))

# --- Thinking budget (Gemini 2.5 es un modelo de "razonamiento híbrido":
# antes de responder, gasta tokens de pensamiento interno que no se ven en
# el output pero sí se facturan y consumen tiempo). Sin configurar nada
# (thinking_budget=-1, "dinámico"), el modelo decide solo cuánto pensar --
# esto es lo que hoy hace que cada llamada tarde ~14-20s. Bajarlo a un
# valor fijo chico reduce esa latencia; en 0 lo apaga del todo. El valor
# por default de acá es un punto de partida conservador ("bajarlo un
# poco", no apagarlo) -- hay que medir el impacto real en tiempo/tokens/
# aciertos con scripts/run_evaluation.py antes de la entrega y ajustar
# este número si hace falta.
LLM_THINKING_BUDGET = int(os.getenv("LLM_THINKING_BUDGET", "512"))

# --- Pesos de la fórmula de imaging_needed_probability ---
URGENCY_WEIGHTS = {
    "alta": 1.0,
    "media": 0.75,
    "baja": 0.5,
}

# --- Umbrales de decisión ---
DERIVAR_THRESHOLD = float(os.getenv("DERIVAR_THRESHOLD", "0.6"))
SEGUIMIENTO_THRESHOLD = float(os.getenv("SEGUIMIENTO_THRESHOLD", "0.3"))

# --- Componente 2: imagen de referencia ilustrativa (Stable Diffusion) ---
# No hay imagen real de paciente en el dataset ni comparación contra ella
# (acordado con la cátedra) -- esto genera una imagen ilustrativa a partir
# del meddiffusion_prompt del GT matcheado, cacheada por gt_id.
# Nihirc/Prompt2MedImage: Stable Diffusion afinado con imágenes médicas
# reales (dataset ROCO) -- mismo formato/API que SD estándar, pero da
# resultados mucho más realistas para este dominio.
STABLE_DIFFUSION_MODEL = os.getenv("STABLE_DIFFUSION_MODEL", "Nihirc/Prompt2MedImage")
REFERENCE_IMAGES_DIR = PROJECT_ROOT / os.getenv("REFERENCE_IMAGES_DIR", "data/generated/reference_images")