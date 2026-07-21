"""
Corre Componente 1 (y Componente 2 cuando corresponde) sobre los casos
clínicos reales del dataset, SIEMPRE desde cero (ignora por completo
cualquier resultado guardado de una corrida anterior, no saltea ningún
caso), y al final imprime (y guarda en evaluation/results/metrics_report.txt)
el reporte completo de métricas (C1, C2 y Sistema Integrado).

Guarda cada resultado en evaluation/results/batch_results.json a medida
que lo procesa (para poder revisar token_usage caso por caso), pero el
archivo queda sobrescrito con SOLO los casos de esta corrida.

Si la cuota DIARIA de la API se agota a mitad de camino (no un error 429
transitorio cualquiera, sino específicamente "se acabaron las llamadas de
hoy"), la corrida corta ahí mismo y reporta las métricas sobre los casos
ya procesados hasta ese punto -- lo ya guardado en el json no se pierde.

Uso:
    python scripts/run_evaluation.py                # corre los 110 casos desde cero
    python scripts/run_evaluation.py --limit 10      # corre solo los primeros 10 casos, para probar rápido
    python scripts/run_evaluation.py --report-only   # no llama al LLM, solo reporta con lo ya guardado
"""

import argparse
import contextlib
import io
import json
import time

from google.genai.errors import ClientError

from oncobridge import config
from oncobridge.component1.pipeline import run_component1
from oncobridge.component2.pipeline import run_component2
from oncobridge.evaluation.metrics_component1 import aggregate_component1_metrics
from oncobridge.evaluation.metrics_component2 import aggregate_component2_metrics
from oncobridge.evaluation.metrics_integrado import (
    compute_triage_time_reduction,
    tasa_imagen_innecesaria,
    time_call,
)
from oncobridge.evaluation.specialist_feedback import read_specialist_feedback_summary
from oncobridge.ingestion.gt_loader import load_ground_truth_base
from oncobridge.llm.client import LLMClient
from oncobridge.rag.retriever import HybridRetriever
from oncobridge.schemas.component1_io import Component1Output, PatientInput
from oncobridge.schemas.component2_io import Component2Output
from oncobridge.schemas.evaluation import ExpectedOutput

RESULTS_PATH = config.PROJECT_ROOT / "evaluation" / "results" / "batch_results.json"
METRICS_REPORT_PATH = config.PROJECT_ROOT / "evaluation" / "results" / "metrics_report.txt"


def _load_results() -> dict:
    if RESULTS_PATH.exists():
        return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    return {}


def _save_results(results: dict) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


def _process_case(case_dir, gt_entries, retriever, client) -> dict:
    patient = PatientInput.model_validate(
        json.loads((case_dir / "input.json").read_text(encoding="utf-8"))
    )

    c1_output, c1_elapsed = time_call(
        run_component1, patient, gt_entries=gt_entries, retriever=retriever, llm_client=client
    )

    entry = {
        "c1_output": c1_output.model_dump(mode="json"),
        "c1_elapsed_seconds": c1_elapsed,
        "c2_output": None,
        "c2_elapsed_seconds": None,
        "error": None,
    }

    if c1_output.recommendation == "DERIVAR_A_IMAGEN" and c1_output.matched_ground_truths:
        c2_output, c2_elapsed = time_call(
            run_component2, c1_output, gt_entries=gt_entries, llm_client=client
        )
        entry["c2_output"] = c2_output.model_dump(mode="json")
        entry["c2_elapsed_seconds"] = c2_elapsed

    return entry


def _message_indicates_daily_quota_exhaustion(message: str) -> bool:
    return "RESOURCE_EXHAUSTED" in message and "PerDay" in message


def _is_daily_quota_exhausted(exc: Exception) -> bool:
    """
    Distingue "se agotó la cuota DIARIA" (no vale la pena seguir
    intentando otros casos, van a fallar igual) de cualquier otro error
    puntual (ese sí solo afecta a este caso, seguimos con el resto).
    """
    if not isinstance(exc, ClientError):
        return False
    return _message_indicates_daily_quota_exhaustion(str(exc))


def _print_timing_summary(
    tiempos_por_caso: list[float],
    tiempo_total: float,
    procesados_esta_corrida: int,
    total_esperados_esta_corrida: int,
    total_casos_dataset: int,
    total_resueltos_acumulado: int,
) -> None:
    """
    Registro de tiempos de ejecución de la corrida: cuántos casos se
    llegaron a procesar, cuántos faltan (de esta corrida y del dataset
    completo), y cuánto tardó en total y por caso -- para llevar la cuenta
    de si el sistema entra en el tope de tiempo de la consigna.
    """
    faltan_esta_corrida = total_esperados_esta_corrida - procesados_esta_corrida
    faltan_dataset = total_casos_dataset - total_resueltos_acumulado
    promedio = tiempo_total / procesados_esta_corrida if procesados_esta_corrida else 0.0

    print("\n" + "-" * 50)
    print("Registro de tiempos de esta corrida")
    print("-" * 50)
    print(f"Casos procesados esta corrida: {procesados_esta_corrida}/{total_esperados_esta_corrida}"
          f" (faltan de esta corrida: {faltan_esta_corrida})")
    print(f"Acumulado del dataset:          {total_resueltos_acumulado}/{total_casos_dataset}"
          f" (faltan en total: {faltan_dataset})")
    print(f"Tiempo total de esta corrida:   {tiempo_total:.1f}s ({tiempo_total / 60:.1f} min)")
    print(f"Tiempo promedio por caso:       {promedio:.1f}s")
    print("-" * 50)


def _run_evaluation(limit: int | None = None) -> None:
    """
    Corre el pipeline completo sobre los casos del dataset, SIEMPRE desde
    cero (ignora por completo lo que haya en batch_results.json de una
    corrida anterior -- no saltea ningún caso por estar ya resuelto).

    Guarda cada resultado en batch_results.json a medida que lo procesa
    (para poder revisar token_usage caso por caso), pero el archivo queda
    sobrescrito con SOLO los casos de esta corrida.

    limit: si se pasa, corta a los primeros N casos -- para probar rápido
    el impacto de un cambio de config sobre una muestra chica, sin esperar
    la corrida completa de los 110.

    Si la cuota diaria se agota a mitad de camino, corta ahí y reporta las
    métricas sobre los casos ya procesados hasta ese punto (lo ya guardado
    en el json queda, no se pierde).
    """
    gt_entries = load_ground_truth_base(config.GT_BASE_DIR)
    retriever = HybridRetriever()
    retriever.build(gt_entries)
    client = LLMClient()

    case_dirs = sorted(config.CLINICAL_CASES_DIR.glob("case_*"))
    if limit:
        case_dirs = case_dirs[:limit]
    results: dict = {}
    tiempos_por_caso: list[float] = []
    inicio_corrida = time.perf_counter()

    for i, case_dir in enumerate(case_dirs, start=1):
        case_id = case_dir.name
        inicio_caso = time.perf_counter()
        try:
            results[case_id] = _process_case(case_dir, gt_entries, retriever, client)
            _save_results(results)
            elapsed_caso = time.perf_counter() - inicio_caso
            tiempos_por_caso.append(elapsed_caso)
            print(f"[{i}/{len(case_dirs)}] {case_id} OK ({elapsed_caso:.1f}s)")
        except Exception as exc:
            elapsed_caso = time.perf_counter() - inicio_caso
            if _is_daily_quota_exhausted(exc):
                print(
                    f"\nCuota diaria de la API agotada en {case_id} (tras {elapsed_caso:.1f}s) -- "
                    f"se corta acá. Reporte con los {len(results)} casos ya procesados:"
                )
                break
            tiempos_por_caso.append(elapsed_caso)
            print(f"[{i}/{len(case_dirs)}] {case_id} ERROR ({elapsed_caso:.1f}s): {exc}")
            results[case_id] = {
                "c1_output": None,
                "c1_elapsed_seconds": None,
                "c2_output": None,
                "c2_elapsed_seconds": None,
                "error": str(exc),
            }
            _save_results(results)

    _print_timing_summary(
        tiempos_por_caso,
        tiempo_total=time.perf_counter() - inicio_corrida,
        procesados_esta_corrida=len(tiempos_por_caso),
        total_esperados_esta_corrida=len(case_dirs),
        total_casos_dataset=len(case_dirs),
        total_resueltos_acumulado=len(results),
    )
    _print_and_save_report(results)


def _load_expected(case_id: str) -> ExpectedOutput:
    raw = json.loads(
        (config.CLINICAL_CASES_DIR / case_id / "expected_output.json").read_text(encoding="utf-8")
    )
    return ExpectedOutput.model_validate(raw)


def _print_report(results: dict | None = None) -> None:
    """
    Si no se pasa `results` (uso de --report-only), lo carga de
    batch_results.json. Si se pasa (uso normal, después de correr los
    casos), reporta directamente sobre ese dict en memoria.
    """
    if results is None:
        results = _load_results()
    total_cases = len(list(config.CLINICAL_CASES_DIR.glob("case_*")))
    gt_entries = load_ground_truth_base(config.GT_BASE_DIR)
    gt_index = {e.gt_id: e for e in gt_entries}

    errored = [case_id for case_id, entry in results.items() if entry.get("error")]
    ok_case_ids = [case_id for case_id in results if case_id not in errored]

    print("\n" + "=" * 50)
    print(" OncoBridge AI — Reporte de Evaluación")
    print("=" * 50)
    print(f"Casos resueltos: {len(results)}/{total_cases}", end="")
    if len(results) < total_cases:
        print(" (corrida incompleta -- se cortó antes de terminar, ver el motivo más arriba)")
    else:
        print()
    if errored:
        print(f"Casos con error (excluidos de las métricas): {len(errored)} -> {errored}")

    c1_pairs: list[tuple[Component1Output, ExpectedOutput]] = []
    c2_pairs: list[tuple[Component2Output, ExpectedOutput]] = []
    total_elapsed_seconds: list[float] = []

    for case_id in ok_case_ids:
        entry = results[case_id]
        expected = _load_expected(case_id)
        c1_output = Component1Output.model_validate(entry["c1_output"])
        c1_pairs.append((c1_output, expected))

        # Tiempo real "con sistema" para este caso: C1 siempre, más C2 cuando
        # el caso efectivamente se derivó a imagen -- si no, el clínico nunca
        # hubiera esperado ese segundo paso, así que no cuenta para el.
        case_elapsed_seconds = entry["c1_elapsed_seconds"]
        if entry["c2_output"] is not None:
            c2_output = Component2Output.model_validate(entry["c2_output"])
            c2_pairs.append((c2_output, expected))
            case_elapsed_seconds += entry["c2_elapsed_seconds"]
        total_elapsed_seconds.append(case_elapsed_seconds)

    if not c1_pairs:
        print("\nTodavía no hay ningún caso resuelto -- corré el script sin --report-only primero.")
        return

    c1_report = aggregate_component1_metrics(c1_pairs)
    print(f"\n--- Componente 1 (N={c1_report.n_cases}) ---")
    print(f"Accuracy de derivación:       {c1_report.accuracy_derivacion:.1%}")
    print(f"Sensibilidad:                 {c1_report.sensibilidad:.1%}")
    print(f"Especificidad:                {c1_report.especificidad:.1%}")
    print(
        f"Precisión de GT match:        {c1_report.precision_gt_match:.1%} "
        f"(sobre {c1_report.n_casos_gt_match_aplicable}/{c1_report.n_cases} casos aplicables)"
    )
    calibracion = c1_report.calibracion_gt_probability
    print(f"Calibración (prob. vs acierto): {calibracion:.2f}" if calibracion is not None else "Calibración:                  sin datos suficientes")
    print(f"Tokens promedio por caso:     {c1_report.tokens_promedio_por_caso:.0f}")
    print("Coherencia del Razonamiento:  ver sección de feedback de especialistas, abajo")

    if c2_pairs:
        c2_report = aggregate_component2_metrics(c2_pairs, gt_index)
        print(f"\n--- Componente 2 (N={c2_report.n_cases}) ---")
        print(f"Sensibilidad de hallazgos:    {c2_report.sensibilidad_hallazgos:.1%}")
        print(f"Especificidad de hallazgos:   {c2_report.especificidad_hallazgos:.1%}")
        print(f"Precisión de Segmentación (IoU proxy): {c2_report.segmentacion_iou_proxy_promedio:.2f}")
        concordancia = c2_report.concordancia_clinica
        print(f"Concordancia clínica:         {concordancia:.2f}" if concordancia is not None else "Concordancia clínica:        sin datos suficientes")
        print("Calidad del Informe:          ver sección de feedback de especialistas, abajo")
    else:
        print("\n--- Componente 2 ---\nNingún caso resuelto todavía llegó a activar Componente 2.")

    print("\n--- Sistema Integrado ---")
    tasa = tasa_imagen_innecesaria(c1_pairs)
    print(f"Tasa de Imagen Innecesaria:   {tasa:.1%}")

    triage = compute_triage_time_reduction(total_elapsed_seconds)
    print("Reducción de Tiempo de Triage (estimada):")
    print(f"  Tiempo con sistema (medido):    {triage.measured_seconds_promedio:.2f} seg/caso promedio (N={len(total_elapsed_seconds)})")
    print(f"  Tiempo sin sistema (referencia): {triage.referencia_manual_minutos:.2f} min")
    print(f"    Fuente: {triage.referencia_fuente}")
    print(f"  Reducción estimada:              ~{triage.reduccion_pct:.1f}%")
    print(
        "  Limitación: la referencia mide revisión de historial en general, no "
        "específica de oncología, y no es un experimento propio con estos pacientes."
    )

    feedback = read_specialist_feedback_summary()
    print("\nSatisfacción del Especialista / Calidad del Informe / Coherencia del Razonamiento:")
    if feedback is None:
        print("  Sin evaluaciones de especialistas registradas aún -- correr scripts/collect_specialist_feedback.py")
    else:
        print(f"  Basado en {feedback['n_respuestas']} respuesta(s):")
        print(f"  NPS promedio:                    {feedback['nps_score_promedio']:.1f} / 5")
        print(f"  Coherencia del razonamiento (C1): {feedback['coherencia_razonamiento_promedio']:.1f} / 5")
        print(f"  Completitud del informe (C2):     {feedback['completitud_informe_promedio']:.1f} / 5")
        print(f"  Precisión del informe (C2):       {feedback['precision_informe_promedio']:.1f} / 5")
        print(f"  Claridad del informe (C2):        {feedback['claridad_informe_promedio']:.1f} / 5")


def _print_and_save_report(results: dict | None = None) -> None:
    """
    Envoltorio de _print_report: además de imprimir en la terminal, guarda
    el mismo texto en evaluation/results/metrics_report.txt -- así el
    reporte no se pierde al cerrar la consola y se puede citar en el
    README sin tener que volver a correr todo el script (que tarda).
    Sobrescribe el archivo en cada corrida (refleja siempre la última).
    """
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        _print_report(results)
    output = buffer.getvalue()

    print(output, end="")

    METRICS_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_REPORT_PATH.write_text(output, encoding="utf-8")
    print(f"\n(Reporte también guardado en {METRICS_REPORT_PATH})")


def main():
    parser = argparse.ArgumentParser(description="Evaluación batch de OncoBridge AI")
    parser.add_argument(
        "--limit", type=int, default=None, help="Máximo de casos a procesar (para probar rápido, ej. 10)"
    )
    parser.add_argument(
        "--report-only", action="store_true", help="No llama al LLM, solo reporta con lo ya guardado"
    )
    args = parser.parse_args()

    if args.report_only:
        _print_and_save_report()
        return

    _run_evaluation(limit=args.limit)


if __name__ == "__main__":
    main()
