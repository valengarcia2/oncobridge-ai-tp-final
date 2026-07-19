"""
Tests del sintetizador de imágenes. El camino de CACHÉ (imagen ya generada)
se prueba sin descargar Stable Diffusion -- corre siempre. La generación
real está cubierta por separado en scripts/generate_reference_images.py,
que se corre manualmente una vez (pesado: baja el modelo, ~4GB).
"""

from oncobridge import config
from oncobridge.component2 import image_synthesizer
from oncobridge.ingestion.gt_loader import load_ground_truth_base


def test_reference_image_path_is_deterministic_by_gt_id():
    path_a = image_synthesizer.reference_image_path("GT-RENAL-001")
    path_b = image_synthesizer.reference_image_path("GT-RENAL-001")
    assert path_a == path_b
    assert path_a.name == "GT-RENAL-001.png"


def test_generate_reference_image_skips_pipeline_when_already_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REFERENCE_IMAGES_DIR", tmp_path)

    entries = load_ground_truth_base(config.GT_BASE_DIR)
    entry = entries[0]

    cached_path = tmp_path / f"{entry.gt_id}.png"
    cached_path.write_bytes(b"fake cached image")

    def _should_not_be_called():
        raise AssertionError("no debería cargar el pipeline si la imagen ya está cacheada")

    monkeypatch.setattr(image_synthesizer, "_get_pipeline", _should_not_be_called)

    result_path = image_synthesizer.generate_reference_image(entry)

    assert result_path == cached_path
    assert result_path.read_bytes() == b"fake cached image"
