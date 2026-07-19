"""
Tests del módulo de persistencia de feedback de especialistas. No prueba
el script interactivo (scripts/collect_specialist_feedback.py) -- eso
necesita un humano tipeando de verdad, igual que run_component1.py/run_e2e.py
no tienen test propio. Acá solo se prueba guardar/leer el JSON.
"""

from oncobridge.evaluation.specialist_feedback import read_specialist_feedback_summary, save_response


def _response(nps=8, coherencia=4, completitud=5, precision=4, claridad=5) -> dict:
    return {
        "nps_score": nps,
        "coherencia_razonamiento": coherencia,
        "completitud_informe": completitud,
        "precision_informe": precision,
        "claridad_informe": claridad,
        "comentario": "",
    }


def test_summary_is_none_when_file_does_not_exist(tmp_path):
    path = tmp_path / "no_existe.json"
    assert read_specialist_feedback_summary(path) is None


def test_save_and_read_single_response(tmp_path):
    path = tmp_path / "feedback.json"
    save_response(_response(nps=8), path=path)

    summary = read_specialist_feedback_summary(path)

    assert summary["n_respuestas"] == 1
    assert summary["nps_score_promedio"] == 8.0
    assert summary["coherencia_razonamiento_promedio"] == 4.0


def test_multiple_responses_accumulate_and_average(tmp_path):
    path = tmp_path / "feedback.json"
    save_response(_response(nps=6), path=path)
    save_response(_response(nps=10), path=path)

    summary = read_specialist_feedback_summary(path)

    assert summary["n_respuestas"] == 2
    assert summary["nps_score_promedio"] == 8.0
