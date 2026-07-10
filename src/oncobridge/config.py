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
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
LLM_MODEL_REASONING = os.getenv("LLM_MODEL_REASONING", "gemini-2.5-flash-lite")
LLM_MODEL_SUMMARIZATION = os.getenv("LLM_MODEL_SUMMARIZATION", "gemini-2.5-flash-lite")
LLM_MODEL_VISION = os.getenv("LLM_MODEL_VISION", "gemini-2.5-flash-lite")

# --- Umbrales de eficiencia de contexto ---
COMPLEX_HISTORY_THRESHOLD = int(os.getenv("COMPLEX_HISTORY_THRESHOLD", "5"))
RETRIEVER_TOP_K = int(os.getenv("RETRIEVER_TOP_K", "5"))
RETRIEVER_MIN_SCORE = float(os.getenv("RETRIEVER_MIN_SCORE", "0.15"))

# --- Pesos de la fórmula de imaging_needed_probability (Paso 8) ---
URGENCY_WEIGHTS = {
    "alta": 1.0,
    "media": 0.75,
    "baja": 0.5,
}

# --- Umbrales de decisión ---
DERIVAR_THRESHOLD = float(os.getenv("DERIVAR_THRESHOLD", "0.6"))
SEGUIMIENTO_THRESHOLD = float(os.getenv("SEGUIMIENTO_THRESHOLD", "0.3"))