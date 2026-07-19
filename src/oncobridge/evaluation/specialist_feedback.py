"""
Persistencia del feedback de especialistas reales (Satisfacción del
Especialista / NPS + Calidad del Informe + Coherencia del Razonamiento --
las 3 métricas que la consigna define como evaluación humana, no
automatizable). Separado a propósito de "collect": este módulo solo
guarda/lee el archivo JSON, no le pregunta nada a nadie -- eso lo hace
scripts/collect_specialist_feedback.py.

El reporte batch (Paso 16) lee read_specialist_feedback_summary() y, si
todavía no hay ninguna respuesta guardada, debe mostrarlo honestamente
("Sin evaluaciones registradas aún") en vez de inventar un promedio.
"""

import json
from pathlib import Path

from oncobridge import config

FEEDBACK_PATH = config.PROJECT_ROOT / "evaluation" / "results" / "specialist_feedback.json"

_SCORE_FIELDS = [
    "nps_score",
    "coherencia_razonamiento",
    "completitud_informe",
    "precision_informe",
    "claridad_informe",
]


def save_response(response: dict, path: Path | None = None) -> None:
    """Agrega una respuesta al archivo de feedback, sumándose a las anteriores si ya había."""
    path = path or FEEDBACK_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    existing = []
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))

    existing.append(response)
    path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")


def read_specialist_feedback_summary(path: Path | None = None) -> dict | None:
    """
    Devuelve un resumen promediado de las respuestas guardadas, o None si
    todavía no hay ninguna -- el llamador (el reporte de Paso 16) es quien
    decide cómo mostrar honestamente la ausencia de datos.
    """
    path = path or FEEDBACK_PATH
    if not path.exists():
        return None

    responses = json.loads(path.read_text(encoding="utf-8"))
    if not responses:
        return None

    n = len(responses)
    return {
        "n_respuestas": n,
        **{f"{field}_promedio": sum(r[field] for r in responses) / n for field in _SCORE_FIELDS},
    }
