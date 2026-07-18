"""
Única puerta de entrada al proveedor LLM (Gemini). Tanto Componente 1 como
Componente 2 pasan por acá.

Usa "salida estructurada" (response_schema con un modelo Pydantic): la API
de Gemini garantiza que la respuesta cumple el schema exacto que le pasamos,
en vez de pedirle "devolvé JSON" en texto libre (más frágil).
"""

from typing import TypeVar
from google import genai
from google.genai import types
from pydantic import BaseModel

from oncobridge import config
from oncobridge.llm.token_tracker import extract_token_usage
from oncobridge.schemas.component1_io import TokenUsage

T = TypeVar("T", bound=BaseModel)


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

        temperature baja (0.2) por default: en un CDSS preferimos
        respuestas consistentes antes que creativas.
        """
        response = self._client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=response_schema,
                temperature=temperature,
            ),
        )

        parsed: T = response.parsed
        token_usage = extract_token_usage(response, model=model)
        return parsed, token_usage