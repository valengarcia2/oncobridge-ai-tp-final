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
"""

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
    ) -> tuple[T, TokenUsage]:
        """
        Llama al modelo pidiendo que la respuesta cumpla exactamente
        `response_schema`. Devuelve (objeto parseado, uso de tokens).
        """
        response = self._generate_with_retry(
            model=model,
            prompt=prompt,
            system_instruction=system_instruction,
            response_schema=response_schema,
            temperature=temperature,
        )

        parsed: T = response.parsed
        token_usage = extract_token_usage(response, model=model)
        return parsed, token_usage

    def _generate_with_retry(self, model, prompt, system_instruction, response_schema, temperature):
        """
        Reintenta hasta _MAX_RETRIES veces si el error es transitorio
        (503/429), con espera creciente entre intento e intento (backoff
        exponencial: 1s, 2s, 4s). Cualquier otro tipo de error se propaga
        inmediatamente, porque reintentar no lo va a arreglar.
        """
        last_error = None
        for attempt in range(_MAX_RETRIES):
            try:
                return self._client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        response_mime_type="application/json",
                        response_schema=response_schema,
                        temperature=temperature,
                    ),
                )
            except (ServerError, ClientError) as e:
                status_code = getattr(e, "code", None)
                is_retryable = status_code in _RETRYABLE_STATUS_CODES
                last_error = e
                if not is_retryable or attempt == _MAX_RETRIES - 1:
                    raise
                wait_time = _BASE_BACKOFF_SECONDS * (2 ** attempt)
                print(
                    f"[LLMClient] Error transitorio ({status_code}) en intento "
                    f"{attempt + 1}/{_MAX_RETRIES}. Reintentando en {wait_time}s..."
                )
                time.sleep(wait_time)
        raise RuntimeError("No se pudo completar la llamada al LLM tras los reintentos") from last_error