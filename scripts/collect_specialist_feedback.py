"""
Recolecta feedback de un especialista real que probó la demo (NPS +
evaluación de calidad del informe/razonamiento). Corré esto cada vez que
un médico prueba el sistema -- las respuestas se van acumulando en
evaluation/results/specialist_feedback.json.

Separado a propósito del script de evaluación batch: ese corre
solo, sobre los 110 casos, sin nadie mirando -- no tiene sentido pedirle a
un médico que responda una encuesta ahí. Esta es la única parte del
proyecto que necesita que un humano se siente a tipear honestamente.

Uso: python scripts/collect_specialist_feedback.py
"""

from oncobridge.evaluation.specialist_feedback import FEEDBACK_PATH, save_response


def _ask_int(prompt: str, min_value: int, max_value: int) -> int:
    while True:
        raw = input(f"{prompt} ({min_value}-{max_value}): ").strip()
        try:
            value = int(raw)
        except ValueError:
            print("  Ingresá un número entero.")
            continue
        if min_value <= value <= max_value:
            return value
        print(f"  Tiene que estar entre {min_value} y {max_value}.")


def collect_one_response() -> dict:
    print("\n=== Encuesta de evaluación por especialista — OncoBridge AI ===\n")

    nps_score = _ask_int(
        "Del 1 al 5, ¿qué tan probable es que recomendarías este sistema a un colega?", 1, 5
    )
    coherencia_razonamiento = _ask_int(
        "Coherencia del razonamiento clínico del Componente 1 (1=malo, 5=excelente)", 1, 5
    )
    completitud_informe = _ask_int(
        "Completitud del informe del Componente 2 (1=malo, 5=excelente)", 1, 5
    )
    precision_informe = _ask_int(
        "Precisión del informe del Componente 2 (1=malo, 5=excelente)", 1, 5
    )
    claridad_informe = _ask_int(
        "Claridad del informe del Componente 2 (1=malo, 5=excelente)", 1, 5
    )
    comentario = input("Comentario libre (opcional, Enter para omitir): ").strip()

    return {
        "nps_score": nps_score,
        "coherencia_razonamiento": coherencia_razonamiento,
        "completitud_informe": completitud_informe,
        "precision_informe": precision_informe,
        "claridad_informe": claridad_informe,
        "comentario": comentario,
    }


def main():
    response = collect_one_response()
    save_response(response)
    print(f"\nGuardado en {FEEDBACK_PATH}. ¡Gracias!")


if __name__ == "__main__":
    main()
