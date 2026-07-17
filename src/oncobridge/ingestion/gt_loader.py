"""
Carga la base de Ground Truth oncológico y la expone como objetos tipados
(GroundTruthEntry), validados contra el schema.
"""

import json
from pathlib import Path

from oncobridge.schemas.ground_truth import GroundTruthEntry


def load_ground_truth_base(path: Path) -> list[GroundTruthEntry]:
    """
    Lee todos los .json de la carpeta oncology_ground_truth_base/ y los
    devuelve como lista de GroundTruthEntry validados.
    """
    entries: list[GroundTruthEntry] = []
    for gt_file in sorted(Path(path).glob("*.json")):
        raw = json.loads(gt_file.read_text(encoding="utf-8"))
        entries.append(GroundTruthEntry.model_validate(raw))
    return entries