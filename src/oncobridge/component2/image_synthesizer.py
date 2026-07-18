"""
Genera (o recupera de caché) la imagen de referencia ilustrativa para una
entrada del ground truth, usando Stable Diffusion a partir de
meddiffusion_prompt / meddiffusion_negative_prompt.

Se cachea en disco por gt_id (no por caso clínico): la imagen depende
únicamente del GT matcheado, así que si el mismo gt_id es la hipótesis
principal en varios de los 110 casos, se genera una sola vez y se reusa.

Limitación honesta a documentar en el README: Stable Diffusion no fue
entrenado específicamente en imágenes médicas -- la imagen generada es una
guía visual ilustrativa para el radiólogo, no un estudio real ni una
simulación clínicamente validada.
"""

from pathlib import Path

from oncobridge import config
from oncobridge.schemas.ground_truth import GroundTruthEntry

_pipeline = None  # cargado perezosamente: bajar los pesos del modelo es costoso


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        import torch
        from diffusers import StableDiffusionPipeline

        _pipeline = StableDiffusionPipeline.from_pretrained(
            config.STABLE_DIFFUSION_MODEL,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        if torch.cuda.is_available():
            _pipeline = _pipeline.to("cuda")
    return _pipeline


def reference_image_path(gt_id: str) -> Path:
    return config.REFERENCE_IMAGES_DIR / f"{gt_id}.png"


def generate_reference_image(entry: GroundTruthEntry, force: bool = False) -> Path:
    """
    Devuelve el path de la imagen de referencia para esta entrada GT,
    generándola con Stable Diffusion si todavía no está cacheada.
    """
    path = reference_image_path(entry.gt_id)
    if path.exists() and not force:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    guidance = entry.radiologist_guidance
    pipeline = _get_pipeline()
    image = pipeline(
        prompt=guidance.meddiffusion_prompt,
        negative_prompt=guidance.meddiffusion_negative_prompt,
    ).images[0]
    image.save(path)
    return path
