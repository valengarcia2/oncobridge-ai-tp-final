"""
Pre-genera la imagen de referencia de las 30 entradas del ground truth con
Stable Diffusion, para no pagar el costo de generación durante la demo ni
durante la corrida de evaluación batch.

Uso: python scripts/generate_reference_images.py
"""

from oncobridge import config
from oncobridge.component2.image_synthesizer import generate_reference_image
from oncobridge.ingestion.gt_loader import load_ground_truth_base


def main():
    entries = load_ground_truth_base(config.GT_BASE_DIR)
    print(f"Generando/verificando {len(entries)} imágenes de referencia en {config.REFERENCE_IMAGES_DIR}...\n")

    for i, entry in enumerate(entries, start=1):
        path = generate_reference_image(entry)
        print(f"[{i}/{len(entries)}] {entry.gt_id} -> {path}")

    print("\nListo.")


if __name__ == "__main__":
    main()
