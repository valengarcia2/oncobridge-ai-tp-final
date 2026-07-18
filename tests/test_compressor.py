"""
Mide, con números reales, cuánto reduce el compressor el tamaño en tokens
de una entrada GT antes de mandarla al LLM.

Nota: usamos tiktoken como PROXY de conteo de tokens. Gemini no usa
exactamente el mismo tokenizador, pero tiktoken es el estándar para estimar 
"aproximadamente cuántos tokens", y sirve para medir la
REDUCCIÓN RELATIVA (completo vs. comprimido), que es lo que nos importa.
"""

import json
import tiktoken

from oncobridge import config
from oncobridge.ingestion.gt_loader import load_ground_truth_base
from oncobridge.rag.compressor import compress_for_reasoning

_encoder = tiktoken.get_encoding("cl100k_base")


def _count_tokens(obj) -> int:
    text = json.dumps(obj, ensure_ascii=False)
    return len(_encoder.encode(text))


def test_compressor_reduces_token_count_significantly():
    entries = load_ground_truth_base(config.GT_BASE_DIR)

    total_full = 0
    total_compressed = 0

    for entry in entries:
        full_tokens = _count_tokens(entry.model_dump(by_alias=True))
        compressed_tokens = _count_tokens(compress_for_reasoning(entry))
        total_full += full_tokens
        total_compressed += compressed_tokens

    reduction_pct = 100 * (1 - total_compressed / total_full)

    print(f"\nTokens totales (30 GT completos):    {total_full}")
    print(f"Tokens totales (30 GT comprimidos):   {total_compressed}")
    print(f"Reducción: {reduction_pct:.1f}%")

    assert reduction_pct > 20, (
        f"Se esperaba una reducción significativa de tokens, se obtuvo solo {reduction_pct:.1f}%"
    )


def test_compressed_entry_excludes_image_generation_fields():
    """Confirma que los campos de imagen quedan afuera del objeto comprimido."""
    entries = load_ground_truth_base(config.GT_BASE_DIR)
    compressed = compress_for_reasoning(entries[0])

    assert "meddiffusion_prompt" not in compressed
    assert "meddiffusion_negative_prompt" not in compressed
    assert "image_generation_notes" not in compressed
    assert "views_recommended" not in compressed

    assert "symptoms" in compressed
    assert "biomarkers" in compressed
    assert "urgency_level" in compressed