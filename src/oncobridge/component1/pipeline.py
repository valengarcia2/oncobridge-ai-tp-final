"""
Pipeline completo de Componente 1: junta retriever, compressor, 
resumen de historial, reasoning engine y
probability engine en un solo flujo, y arma el Component1Output
final tal como lo exige el contrato.

Las `radiologist_instructions` de cada hipótesis matcheada se arman a
partir del GroundTruthEntry ORIGINAL (no las regenera el LLM). Esto evita
que el LLM alucine prompts de MedDiffusion o instrucciones de imagen. 
"""

from oncobridge import config
from oncobridge.ingestion.gt_loader import load_ground_truth_base
from oncobridge.rag.retriever import HybridRetriever
from oncobridge.rag.compressor import compress_for_reasoning
from oncobridge.component1.history_summarizer import summarize_if_needed
from oncobridge.component1.reasoning_engine import reason
from oncobridge.component1.probability_engine import (
    compute_imaging_needed_probability,
    HypothesisForProbability,
)
from oncobridge.llm.client import LLMClient
from oncobridge.schemas.ground_truth import GroundTruthEntry
from oncobridge.schemas.component1_io import (
    PatientInput,
    Component1Output,
    MatchedHypothesis,
    RadiologistInstructions,
    TokenUsage,
)


def _sum_token_usage(usages: list[TokenUsage | None], model: str) -> TokenUsage:
    """Combina el uso de tokens de varias llamadas en un solo TokenUsage."""
    real_usages = [u for u in usages if u is not None]
    return TokenUsage(
        prompt_tokens=sum(u.prompt_tokens for u in real_usages),
        completion_tokens=sum(u.completion_tokens for u in real_usages),
        total_tokens=sum(u.total_tokens for u in real_usages),
        model=model,
    )


def _build_radiologist_instructions(entry: GroundTruthEntry, match_rationale: str) -> RadiologistInstructions:
    """Arma las instrucciones para el radiólogo a partir del GT original (sin LLM)."""
    guidance = entry.radiologist_guidance
    return RadiologistInstructions(
        suggested_modalities=guidance.modality_priority,
        imaging_location=guidance.imaging_location,
        clinical_context_for_radiologist=match_rationale,
        meddiffusion_reference_prompt=guidance.meddiffusion_prompt,
        meddiffusion_negative_prompt=guidance.meddiffusion_negative_prompt,
        reference_images_note=(
            f"Imágenes de referencia MedDiffusion para {entry.gt_id}. "
            "Usar como orientación visual, no como diagnóstico."
        ),
    )


def run_component1(
    patient: PatientInput,
    gt_entries: list[GroundTruthEntry] | None = None,
    retriever: HybridRetriever | None = None,
    llm_client: LLMClient | None = None,
) -> Component1Output:
    """
    Corre el flujo completo de Componente 1 sobre un paciente.
    gt_entries y retriever son opcionales para permitir pasarlos ya
    construidos (evita recargar/reindexar en cada llamada, útil para la
    evaluación batch sobre 110 casos).
    """
    if gt_entries is None:
        gt_entries = load_ground_truth_base(config.GT_BASE_DIR)
    entries_by_id = {e.gt_id: e for e in gt_entries}

    if retriever is None:
        retriever = HybridRetriever()
        retriever.build(gt_entries)

    client = llm_client or LLMClient()

    history_text, history_token_usage = summarize_if_needed(
        patient.medical_history, llm_client=client
    )

    top_candidates = retriever.retrieve(
        patient, top_k=config.RETRIEVER_TOP_K, min_score=config.RETRIEVER_MIN_SCORE
    )
    compressed_candidates = [
        compress_for_reasoning(entries_by_id[gt_id]) for gt_id, _score in top_candidates
    ]
    raw_output, reasoning_token_usage = reason(
        patient, history_text, compressed_candidates, llm_client=client
    )

    hypotheses_for_probability = [
        HypothesisForProbability(
            match_probability=h.match_probability,
            urgency_level=entries_by_id[h.gt_id].urgency_level,
        )
        for h in raw_output.matched_hypotheses
    ]
    imaging_needed_probability, recommendation, urgency = compute_imaging_needed_probability(
        hypotheses_for_probability,
        conclusive=raw_output.conclusive,
        imaging_needed_without_match=raw_output.imaging_needed_without_match,
        no_match_urgency=raw_output.no_match_urgency,
    )

    matched_ground_truths = [
        MatchedHypothesis(
            gt_id=h.gt_id,
            icd_10=entries_by_id[h.gt_id].icd_10,
            icd_10_description=entries_by_id[h.gt_id].icd_10_description,
            match_probability=h.match_probability,
            match_rationale=h.match_rationale,
            radiologist_instructions=_build_radiologist_instructions(
                entries_by_id[h.gt_id], h.match_rationale
            ),
        )
        for h in sorted(raw_output.matched_hypotheses, key=lambda h: h.match_probability, reverse=True)
    ]

    token_usage = _sum_token_usage(
        [history_token_usage, reasoning_token_usage], model=config.LLM_MODEL_REASONING
    )
    token_usage.retrieved_gt_entries = len(top_candidates)
    token_usage.gt_entries_in_context = len(compressed_candidates)

    return Component1Output(
        patient_id=patient.patient_id,
        clinical_summary=raw_output.clinical_summary,
        matched_ground_truths=matched_ground_truths,
        imaging_needed_probability=imaging_needed_probability,
        reasoning=raw_output.reasoning,
        recommendation=recommendation,
        urgency=urgency,
        conclusive=raw_output.conclusive,
        token_usage=token_usage,
    )