"""
Única puerta de entrada al proveedor LLM (Gemini). Tanto Componente 1 como
Componente 2 pasan por acá.

Usa "salida estructurada" (response_schema con un modelo Pydantic): la API
de Gemini garantiza que la respuesta cumple el schema exacto que le pasamos.

Incluye reintento automático con backoff exponencial para errores
transitorios del servidor (503 "sobrecargado", 429 "demasiadas requests").
Esto es necesario en la práctica: la API pública de Gemini tiene picos de
demanda frecuentes, y sin esto cualquier corrida batch sobre los 110 casos
del dataset fallaría espuriamente cada tanto.

El free tier de Gemini tiene DOS cuotas distintas que devuelven el mismo
código 429 RESOURCE_EXHAUSTED, y hay que tratarlas distinto:
- "PerDay" (llamadas por día): no vale la pena reintentar, va a seguir
  fallando hasta que se resetee -- se propaga de inmediato para que el
  script de evaluación corte toda la corrida ahí.
- "PerMinute" (llamadas por minuto, ej. 5 req/min en este proyecto): SÍ
  vale la pena esperar y reintentar -- la propia respuesta de la API
  indica cuántos segundos esperar (retryDelay). Sin esto, una corrida
  batch entera en cascada por cada caso que cae justo en la ventana del
  minuto, marcando casos como error cuando en realidad solo había que
  esperar unos segundos.
"""

import re
import time
from typing import TypeVar
from google import genai
from google.genai import types
from google.genai.errors import ServerError, ClientError
from pydantic import BaseModel

from oncobridge import config
from oncobridge.llm.token_tracker import extract_token_usage
from oncobridge.schemas.component1_io import TokenUsage

T = TypeVar("T", bound=BaseModel)

# Códigos de error que consideramos transitorios (vale la pena reintentar).
_RETRYABLE_STATUS_CODES = {429, 503}
_MAX_RETRIES = 3
_BASE_BACKOFF_SECONDS = 1.0

# Reintentos específicos para el límite de requests-POR-MINUTO (distinto de
# _MAX_RETRIES): acá sí vale la pena esperar bastante más, porque la ventana
# se libera sola en un minuto o menos.
_MAX_RATE_LIMIT_WAITS = 8
_DEFAULT_RATE_LIMIT_WAIT_SECONDS = 30.0
_RATE_LIMIT_WAIT_BUFFER_SECONDS = 2.0


def _is_per_day_quota_error(message: str) -> bool:
    return "RESOURCE_EXHAUSTED" in message and "PerDay" in message


def _is_per_minute_rate_limit_error(message: str) -> bool:
    return "RESOURCE_EXHAUSTED" in message and "PerMinute" in message


def _extract_retry_delay_seconds(message: str) -> float:
    """Lee el 'retryDelay': '34s' que la API sugiere, si está presente."""
    match = re.search(r"retryDelay['\"]?\s*:\s*['\"](\d+(?:\.\d+)?)s", message)
    if match:
        return float(match.group(1)) + _RATE_LIMIT_WAIT_BUFFER_SECONDS
    return _DEFAULT_RATE_LIMIT_WAIT_SECONDS


class LLMClient:
    def __init__(self, api_key: str | None = None):
        self._client = genai.Client(api_key=api_key or config.GEMINI_API_KEY)

    def complete_structured(
        self,
        prompt: str,
        response_schema: type[T],
        model: str,
        system_instruction: str | None = None,
        temperature: float = 0.2,
        thinking_budget: int | None = None,
    ) -> tuple[T, TokenUsage]:
        """
        Llama al modelo pidiendo que la respuesta cumpla exactamente
        `response_schema`. Devuelve (objeto parseado, uso de tokens).

        thinking_budget (tokens de "pensamiento" interno del modelo antes
        de responder): si no se pasa, usa config.LLM_THINKING_BUDGET.
        """
        response = self._generate_with_retry(
            model=model,
            prompt=prompt,
            system_instruction=system_instruction,
            response_schema=response_schema,
            temperature=temperature,
            thinking_budget=(
                thinking_budget if thinking_budget is not None else config.LLM_THINKING_BUDGET
            ),
        )

        parsed: T = response.parsed
        token_usage = extract_token_usage(response, model=model)
        return parsed, token_usage

    def _generate_with_retry(
        self, model, prompt, system_instruction, response_schema, temperature, thinking_budget
    ):
        """
        Reintenta hasta _MAX_RETRIES veces si el error es transitorio
        (503/429 genérico), con backoff exponencial corto (1s, 2s, 4s).

        El 429 de límite POR MINUTO se maneja aparte, con su propio
        contador (_MAX_RATE_LIMIT_WAITS) y esperando lo que la API sugiere
        (retryDelay) -- no cuenta contra _MAX_RETRIES, porque no es un
        error transitorio cualquiera, es una espera esperada y recuperable.

        El 429 de cuota DIARIA no entra en ninguno de los dos casos
        especiales: cae en el backoff genérico y termina propagándose
        (attempt == _MAX_RETRIES - 1) para que el script de evaluación lo
        detecte y corte toda la corrida.
        """
        rate_limit_waits = 0
        attempt = 0
        while True:
            try:
                return self._client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        response_mime_type="application/json",
                        response_schema=response_schema,
                        temperature=temperature,
                        thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
                    ),
                )
            except (ServerError, ClientError) as e:
                status_code = getattr(e, "code", None)
                message = str(e)

                if (
                    status_code == 429
                    and not _is_per_day_quota_error(message)
                    and _is_per_minute_rate_limit_error(message)
                ):
                    if rate_limit_waits >= _MAX_RATE_LIMIT_WAITS:
                        raise
                    rate_limit_waits += 1
                    wait_time = _extract_retry_delay_seconds(message)
                    print(
                        f"[LLMClient] Límite de requests por minuto alcanzado "
                        f"(espera {rate_limit_waits}/{_MAX_RATE_LIMIT_WAITS}). "
                        f"Esperando {wait_time:.0f}s antes de reintentar..."
                    )
                    time.sleep(wait_time)
                    continue

                is_retryable = status_code in _RETRYABLE_STATUS_CODES
                if not is_retryable or attempt >= _MAX_RETRIES - 1:
                    raise
                wait_time = _BASE_BACKOFF_SECONDS * (2 ** attempt)
                attempt += 1
                print(
                    f"[LLMClient] Error transitorio ({status_code}) en intento "
                    f"{attempt}/{_MAX_RETRIES}. Reintentando en {wait_time}s..."
                )
                time.sleep(wait_time)