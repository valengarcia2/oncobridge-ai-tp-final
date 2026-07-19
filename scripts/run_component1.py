"""
CLI para correr Componente 1 sobre un caso clínico del dataset, de punta a
punta. Requiere GEMINI_API_KEY configurada en .env.

Uso: python scripts/run_component1.py --case case_001
"""

import argparse
import json

from oncobridge import config
from oncobridge.ingestion.gt_loader import load_ground_truth_base
from oncobridge.rag.retriever import HybridRetriever
from oncobridge.component1.pipeline import run_component1
from oncobridge.schemas.component1_io import PatientInput


def main():
    parser = argparse.ArgumentParser(description="Correr Componente 1 sobre un caso clínico")
    parser.add_argument("--case", required=True, help="ID del caso, ej. case_001")
    args = parser.parse_args()

    case_dir = config.CLINICAL_CASES_DIR / args.case
    input_path = case_dir / "input.json"
    if not input_path.exists():
        raise SystemExit(f"No se encontró {input_path}")

    print(f"Cargando base de conocimiento GT ({config.GT_BASE_DIR})...")
    gt_entries = load_ground_truth_base(config.GT_BASE_DIR)
    retriever = HybridRetriever()
    retriever.build(gt_entries)

    input_raw = json.loads(input_path.read_text(encoding="utf-8"))
    patient = PatientInput.model_validate(input_raw)

    print(f"Corriendo Componente 1 para {args.case} (paciente {patient.patient_id})...\n")
    output = run_component1(patient, gt_entries=gt_entries, retriever=retriever)

    print(json.dumps(output.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()