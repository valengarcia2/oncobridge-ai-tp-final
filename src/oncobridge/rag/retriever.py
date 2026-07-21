"""
Retriever híbrido: combina un score léxico (BM25) con un score semántico
(embeddings) para rankear las 30 entradas GT contra un paciente dado.
"""

from rank_bm25 import BM25Okapi

from oncobridge.schemas.ground_truth import GroundTruthEntry
from oncobridge.schemas.component1_io import PatientInput
from oncobridge.ingestion.embeddings_index import (
    EmbeddingIndex,
    gt_entry_to_text,
    patient_input_to_text,
)


def _tokenize(text: str) -> list[str]:
    """Tokenización simple: minúsculas + separar por espacios."""
    return text.lower().split()


class HybridRetriever:
    def __init__(self, bm25_weight: float = 0.5, embedding_weight: float = 0.5):
        self.bm25_weight = bm25_weight
        self.embedding_weight = embedding_weight

        self._gt_ids: list[str] = []
        self._bm25: BM25Okapi | None = None
        self._embedding_index = EmbeddingIndex()

    def build(self, entries: list[GroundTruthEntry]) -> None:
        """Se llama UNA vez (offline), no en cada consulta."""
        self._gt_ids = [e.gt_id for e in entries]

        tokenized_docs = [_tokenize(gt_entry_to_text(e)) for e in entries]
        self._bm25 = BM25Okapi(tokenized_docs)

        self._embedding_index.build(entries)

    def _patient_query_text(self, patient: PatientInput) -> str:
        labs_summary = " ".join(f"{k} {v}" for k, v in patient.current_labs.items())
        return patient_input_to_text(
            current_symptoms=patient.current_symptoms,
            family_history=patient.demographics.family_history,
            labs_summary=labs_summary,
        )

    def retrieve(
        self, patient: PatientInput, top_k: int = 5, min_score: float = 0.0
    ) -> list[tuple[str, float]]:
        """
        Devuelve una lista de (gt_id, score_combinado) ordenada de mayor a
        menor, con como máximo top_k elementos.

        min_score descarta candidatos por debajo de ese score combinado
        ANTES de tomar el top_k -- por default 0.0 (sin filtro, se
        devuelven siempre top_k, el comportamiento histórico). El pipeline
        real le pasa config.RETRIEVER_MIN_SCORE: sin este filtro, un
        paciente sin ningún candidato realmente parecido igual recibía
        top_k candidatos irrelevantes rellenando la lista -- más tokens y
        tiempo en el prompt del LLM sin aportar señal real.
        """
        if self._bm25 is None:
            raise RuntimeError("Llamar a build() antes de retrieve()")

        query_text = self._patient_query_text(patient)

        bm25_raw_scores = self._bm25.get_scores(_tokenize(query_text))
        max_bm25 = max(bm25_raw_scores) if max(bm25_raw_scores) > 0 else 1.0
        bm25_scores = {gt_id: s / max_bm25 for gt_id, s in zip(self._gt_ids, bm25_raw_scores)}

        embedding_scores = self._embedding_index.similarity_scores(query_text)

        combined = {
            gt_id: self.bm25_weight * bm25_scores[gt_id]
            + self.embedding_weight * embedding_scores[gt_id]
            for gt_id in self._gt_ids
        }

        ranked = sorted(combined.items(), key=lambda item: item[1], reverse=True)
        ranked = [(gt_id, score) for gt_id, score in ranked if score >= min_score]
        return ranked[:top_k]