from oncobridge import config
from oncobridge.ingestion.gt_loader import load_ground_truth_base


def test_load_ground_truth_base_returns_30_typed_entries():
    entries = load_ground_truth_base(config.GT_BASE_DIR)
    assert len(entries) == 30

    ids = [e.gt_id for e in entries]
    assert len(ids) == len(set(ids)), "gt_id duplicados"

    sample = entries[0]
    assert hasattr(sample, "objective_data")
    assert hasattr(sample, "radiologist_guidance")
    assert sample.urgency_level in ("alta", "media", "baja")