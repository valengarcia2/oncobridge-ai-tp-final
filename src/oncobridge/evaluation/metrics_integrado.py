"""
Métricas de "Sistema Integrado" -- evalúan el comportamiento del sistema completo, no un
componente aislado. Viven en su propio módulo, separado de
metrics_component1.py/metrics_component2.py, aunque reutilicen funciones
de ahí -- son una lectura distinta de los mismos datos, no una propiedad
de Componente 1 en particular.

De las 3 métricas de esa tabla:
- Tasa de Imagen Innecesaria: sí se puede calcular con este dataset (ver
  tasa_imagen_innecesaria() abajo) -- se deriva de los mismos TP/FP de
  derivación a imagen que ya calcula metrics_component1.
- Reducción de Tiempo de Triage: se mide el "tiempo con sistema" de verdad
  (cronometrando run_component1/run_component2 sobre casos reales, ver
  time_call()) y se compara contra una referencia citada de la literatura
  para "tiempo sin sistema" (ver compute_triage_time_reduction()) -- la
  consigna aclara "(benchmark de referencia)", no un ensayo clínico
  propio con los mismos pacientes.
- Satisfacción del Especialista: encuesta real (NPS), no automatizable --
  ver oncobridge.evaluation.specialist_feedback y
  scripts/collect_specialist_feedback.py.
"""

import time
from dataclasses import dataclass

from oncobridge.evaluation.metrics_component1 import imaging_confusion_bucket
from oncobridge.schemas.component1_io import Component1Output
from oncobridge.schemas.evaluation import ExpectedOutput

# Overhage JM, McCallie D Jr. "Physician Time Spent Using the Electronic
# Health Record During Outpatient Encounters: A Descriptive Study."
# Annals of Internal Medicine. 2020;172:169-174. 155.000 medicos en EE.UU.,
# practica ambulatoria con EHR Cerner Millennium: 16 min 14 seg promedio
# por consulta, de los cuales revision de historial clinico = 33%.
_REFERENCIA_MINUTOS_TOTALES_POR_CONSULTA = 16 + 14 / 60
_REFERENCIA_PORCENTAJE_REVISION_HISTORIAL = 0.33
REFERENCIA_TRIAGE_MANUAL_MINUTOS = round(
    _REFERENCIA_MINUTOS_TOTALES_POR_CONSULTA * _REFERENCIA_PORCENTAJE_REVISION_HISTORIAL, 2
)  # ~5.36 minutos
REFERENCIA_TRIAGE_FUENTE = (
    "Overhage JM, McCallie D Jr. Physician Time Spent Using the Electronic "
    "Health Record During Outpatient Encounters: A Descriptive Study. "
    "Annals of Internal Medicine. 2020;172:169-174. Revision de historial "
    "clinico = 33% de 16 min 14 seg promedio por consulta ambulatoria "
    "(155.000 medicos en EE.UU., EHR Cerner Millennium)."
)


def tasa_imagen_innecesaria(cases: list[tuple[Component1Output, ExpectedOutput]]) -> float:
    """
    % de las imágenes que el sistema pidió (recommendation=DERIVAR_A_IMAGEN)
    que resultaron innecesarias (Falsos Positivos: no hacía falta imagen
    igual). Reutiliza imaging_confusion_bucket de metrics_component1 --
    mismo dato de fondo, otra lectura (acá el foco es el costo/carga que
    el sistema le agrega al servicio de imágenes, no su sensibilidad).
    """
    if not cases:
        raise ValueError("Se necesita al menos un caso para calcular esta métrica")

    buckets = [imaging_confusion_bucket(output, expected) for output, expected in cases]
    true_positives = sum(b.true_positive for b in buckets)
    false_positives = sum(b.false_positive for b in buckets)

    total_solicitadas = true_positives + false_positives
    if total_solicitadas == 0:
        return float("nan")
    return false_positives / total_solicitadas


def time_call(fn, *args, **kwargs):
    """
    Cronometra una llamada real (ej. run_component1 sobre un caso) y
    devuelve (resultado, segundos_transcurridos). Pensado para usarse en
    el runner batch, acumulando los tiempos de los 110 casos
    para pasárselos a compute_triage_time_reduction().
    """
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


@dataclass
class TriageTimeReport:
    measured_seconds_promedio: float
    referencia_manual_minutos: float
    referencia_fuente: str
    reduccion_pct: float


def compute_triage_time_reduction(
    measured_seconds: list[float],
    referencia_manual_minutos: float = REFERENCIA_TRIAGE_MANUAL_MINUTOS,
    referencia_fuente: str = REFERENCIA_TRIAGE_FUENTE,
) -> TriageTimeReport:
    """
    Compara el tiempo medido REAL del sistema (measured_seconds, en
    segundos, cronometrado con time_call() sobre casos reales) contra una
    referencia de tiempo de revisión manual de historial, citada de la
    literatura clínica -- no medida empíricamente por el equipo.

    Limitación a documentar en el README: la referencia (default: Overhage
    & McCallie, 2020) mide revisión de historial en general, no específica
    de oncología, y no es un experimento controlado con los mismos
    pacientes de este dataset -- es una comparación contra literatura
    publicada, tal como la propia consigna anticipa con "(benchmark de
    referencia)".
    """
    if not measured_seconds:
        raise ValueError("Se necesita al menos una medición de tiempo")
    if not referencia_fuente:
        raise ValueError("La referencia de tiempo manual necesita una fuente citada")

    promedio_segundos = sum(measured_seconds) / len(measured_seconds)
    promedio_minutos = promedio_segundos / 60

    reduccion_pct = (
        (referencia_manual_minutos - promedio_minutos) / referencia_manual_minutos * 100
        if referencia_manual_minutos > 0
        else float("nan")
    )

    return TriageTimeReport(
        measured_seconds_promedio=promedio_segundos,
        referencia_manual_minutos=referencia_manual_minutos,
        referencia_fuente=referencia_fuente,
        reduccion_pct=reduccion_pct,
    )
