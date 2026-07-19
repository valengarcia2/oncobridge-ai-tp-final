"""
CLI para correr el sistema completo (Componente 1 -> Componente 2) sobre un
caso clínico del dataset, de punta a punta. Requiere GEMINI_API_KEY
configurada en .env.

Si C1 no concluye una hipótesis con recomendación DERIVAR_A_IMAGEN, no se
corre Componente 2 -- replica el flujo clínico real (no se genera ni se lee
una imagen si C1 decidió que no hace falta).

Uso: python scripts/run_e2e.py --case case_002
"""

import argparse
import json

from oncobridge import config
from oncobridge.component1.pipeline import run_component1
from oncobridge.component2.image_synthesizer import reference_image_path
from oncobridge.component2.pipeline import run_component2
from oncobridge.ingestion.gt_loader import load_ground_truth_base
from oncobridge.rag.retriever import HybridRetriever
from oncobridge.schemas.component1_io import PatientInput


def main():
    parser = argparse.ArgumentParser(description="Correr C1 -> C2 sobre un caso clínico")
    parser.add_argument("--case", required=True, help="ID del caso, ej. case_002")
    args = parser.parse_args()

    case_dir = config.CLINICAL_CASES_DIR / args.case
    input_path = case_dir / "input.json"
    if not input_path.exists():
        raise SystemExit(f"No se encontró {input_path}")

    print(f"Cargando base de conocimiento GT ({config.GT_BASE_DIR})...")
    gt_entries = load_ground_truth_base(config.GT_BASE_DIR)
    retriever = HybridRetriever()
    retriever.build(gt_entries)

    patient = PatientInput.model_validate(json.loads(input_path.read_text(encoding="utf-8")))

    print(f"\n=== Componente 1: {args.case} (paciente {patient.patient_id}) ===\n")
    c1_output = run_component1(patient, gt_entries=gt_entries, retriever=retriever)
    print(json.dumps(c1_output.model_dump(), ensure_ascii=False, indent=2))

    if c1_output.recommendation != "DERIVAR_A_IMAGEN" or not c1_output.matched_ground_truths:
        print(
            f"\nC1 no recomendó derivar a imagen (recommendation={c1_output.recommendation}); "
            "no se corre Componente 2."
        )
        return

    print("\n=== Componente 2 ===\n")
    c2_output = run_component2(c1_output, gt_entries=gt_entries)
    print(json.dumps(c2_output.model_dump(), ensure_ascii=False, indent=2))

    top_gt_id = c1_output.matched_ground_truths[0].gt_id
    print(f"\nImagen de referencia generada/cacheada en: {reference_image_path(top_gt_id)}")


if __name__ == "__main__":
    main()
