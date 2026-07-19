"""
CLI para correr Componente 2 sobre un caso clínico del dataset. Requiere
GEMINI_API_KEY configurada en .env.

Componente 2 no puede correr aislado de datos crudos: su input es el
output de Componente 1 (no un estudio de imagen real -- no existe en el
dataset, acordado así con la cátedra). Este script corre C1 internamente
para producir ese input, pero imprime solo el output de C2.

Uso: python scripts/run_component2.py --case case_002
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
    parser = argparse.ArgumentParser(description="Correr Componente 2 sobre un caso clínico")
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

    print("Corriendo Componente 1 primero, para obtener el input que necesita C2...")
    c1_output = run_component1(patient, gt_entries=gt_entries, retriever=retriever)

    if c1_output.recommendation != "DERIVAR_A_IMAGEN" or not c1_output.matched_ground_truths:
        raise SystemExit(
            f"{args.case}: C1 no recomendó derivar a imagen "
            f"(recommendation={c1_output.recommendation}) -- Componente 2 no aplica a este caso. "
            "Probá con otro case_id (ej. case_002)."
        )

    print(f"Corriendo Componente 2 para {args.case} (paciente {c1_output.patient_id})...\n")
    c2_output = run_component2(c1_output, gt_entries=gt_entries)

    print(json.dumps(c2_output.model_dump(), ensure_ascii=False, indent=2))

    top_gt_id = c1_output.matched_ground_truths[0].gt_id
    print(f"\nImagen de referencia: {reference_image_path(top_gt_id)}")


if __name__ == "__main__":
    main()
