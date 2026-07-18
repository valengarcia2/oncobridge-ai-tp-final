"""
Script de humo para confirmar que la conexión con Gemini funciona.
NO es un test de pytest a propósito: necesita API key real y conexión a
internet, y no queremos que eso sea un requisito para correr `pytest tests/`
en cualquier máquina del equipo.

Uso: python scripts/test_llm_connection.py
"""

from pydantic import BaseModel

from oncobridge import config
from oncobridge.llm.client import LLMClient


class SmokeTestResponse(BaseModel):
    status: str
    received_prompt: bool


def main():
    print(f"Usando modelo: {config.LLM_MODEL_REASONING}")
    client = LLMClient()

    parsed, token_usage = client.complete_structured(
        prompt="Responde con status='ok' y received_prompt=true.",
        response_schema=SmokeTestResponse,
        model=config.LLM_MODEL_REASONING,
    )

    print("Respuesta parseada:", parsed)
    print("Token usage:", token_usage)

    assert parsed.status == "ok"
    print("\n Conexión con Gemini funcionando correctamente.")


if __name__ == "__main__":
    main()