"""
Convierte el `usage_metadata` que devuelve la API de Gemini en nuestro
propio schema TokenUsage, de forma uniforme para cualquier llamada.
"""

from oncobridge.schemas.component1_io import TokenUsage


def extract_token_usage(response, model: str) -> TokenUsage:
    usage = response.usage_metadata
    return TokenUsage(
        prompt_tokens=usage.prompt_token_count or 0,
        completion_tokens=usage.candidates_token_count or 0,
        total_tokens=usage.total_token_count or 0,
        model=model,
    )