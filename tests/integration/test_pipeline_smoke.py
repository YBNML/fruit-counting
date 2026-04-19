from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
CFG = ROOT / "configs" / "pipeline" / "default.yaml"


def _weights_available() -> bool:
    try:
        import yaml
    except ImportError:
        return False
    cfg = yaml.safe_load(CFG.read_text())
    stages = cfg["pipeline"]["stages"]
    need = []
    if stages["pseco"]["enabled"]:
        need += [
            stages["pseco"]["sam_checkpoint"],
            stages["pseco"]["decoder_checkpoint"],
            stages["pseco"]["mlp_checkpoint"],
        ]
    if stages["deblur"]["enabled"]:
        need.append(stages["deblur"]["weights"])
    return all((ROOT / p).exists() for p in need)


@pytest.mark.slow
@pytest.mark.skipif(not _weights_available(), reason="model weights not present")
def test_pipeline_runs_on_dummy_image(tmp_path):
    from counting.config.loader import load_config
    from counting.pipeline import build_pipeline

    img_path = tmp_path / "dummy.png"
    Image.fromarray((np.random.rand(256, 256, 3) * 255).astype("uint8")).save(img_path)

    cfg = load_config(CFG, overrides=["device=cpu"])
    pipe = build_pipeline(cfg)
    arr = np.asarray(Image.open(img_path).convert("RGB"))
    r = pipe.run_numpy(arr, image_path=str(img_path))

    assert r.config_hash
    assert r.error is None or "pseco" in r.error
