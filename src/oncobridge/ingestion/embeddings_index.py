"""
Construye las representaciones necesarias para el retriever híbrido:
1. Un "documento" de texto por cada entrada GT (para BM25 y para embeddings).
2. Los embeddings (vectores numéricos) de esos documentos, usando un modelo
   pre-entrenado multilingüe (el dataset está en español).
"""

from sentence_transformers import SentenceTransformer
import numpy as np

from oncobridge.schemas.ground_truth import GroundTruthEntry

# Modelo multilingüe liviano (~470MB), soporta español entre ~50 idiomas.
# La primera vez que se usa, se descarga automáticamente y queda cacheado
# en disco (no se vuelve a descargar en corridas siguientes).
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def gt_entry_to_text(entry: GroundTruthEntry) -> str:
    """
    Convierte una entrada GT en un solo bloque de texto, concatenando los
    campos que sirven para MATCHEAR contra un paciente (síntomas, hallazgos,
    factores de riesgo, frases del paciente). Deliberadamente NO incluye
    campos que son solo para construir el output final (meddiffusion_prompt,
    image_generation_notes) — eso es responsabilidad del compressor,
    en un paso posterior.
    """
    parts = [
        entry.icd_10_description,
        " ".join(entry.subjective_data.symptoms),
        " ".join(entry.subjective_data.patient_reported_concerns),
        entry.subjective_data.onset_pattern,
        " ".join(entry.objective_data.clinical_findings),
        " ".join(entry.objective_data.risk_factors),
        " ".join(f"{k} {v}" for k, v in entry.objective_data.biomarkers.items()),
    ]
    return " . ".join(p for p in parts if p)


def patient_input_to_text(current_symptoms: list[str], family_history: list[str],
                           labs_summary: str = "") -> str:
    """
    Misma idea que gt_entry_to_text, pero para el lado del paciente.
    """
    parts = [
        " ".join(current_symptoms),
        " ".join(family_history),
        labs_summary,
    ]
    return " . ".join(p for p in parts if p)


class EmbeddingIndex:
    """
    Envoltorio simple sobre el modelo de embeddings: calcula y guarda en
    memoria los vectores de las 30 entradas GT una sola vez (build), y
    después permite comparar un texto nuevo (un paciente) contra todos
    ellos con similitud coseno.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.gt_ids: list[str] = []
        self.gt_vectors: np.ndarray | None = None

    def build(self, entries: list[GroundTruthEntry]) -> None:
        texts = [gt_entry_to_text(e) for e in entries]
        self.gt_ids = [e.gt_id for e in entries]
        self.gt_vectors = self.model.encode(texts, normalize_embeddings=True)

    def similarity_scores(self, query_text: str) -> dict[str, float]:
        """Devuelve {gt_id: score_coseno} para el texto de consulta dado."""
        if self.gt_vectors is None:
            raise RuntimeError("Llamar a build() antes de similarity_scores()")
        query_vector = self.model.encode([query_text], normalize_embeddings=True)[0]
        scores = self.gt_vectors @ query_vector
        return dict(zip(self.gt_ids, scores.tolist()))