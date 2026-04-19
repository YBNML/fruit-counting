import textwrap

import pytest

from counting.config.loader import apply_overrides, load_config


def _write(tmp_path, text: str):
    p = tmp_path / "cfg.yaml"
    p.write_text(textwrap.dedent(text).lstrip(), encoding="utf-8")
    return p


def test_load_valid_yaml(tmp_path):
    p = _write(
        tmp_path,
        """
        device: cpu
        seed: 7
        output_dir: ./out
        pipeline:
          stages:
            deblur: {enabled: false}
            pseco:
              enabled: true
              sam_checkpoint: a.pth
              decoder_checkpoint: b.pth
              mlp_checkpoint: c.tar
        """,
    )
    cfg = load_config(p)
    assert cfg.device == "cpu"
    assert cfg.seed == 7
    assert cfg.pipeline.stages.pseco.sam_checkpoint == "a.pth"


def test_load_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "no.yaml")


def test_apply_overrides_bool_and_number():
    cfg = {"pipeline": {"stages": {"sr": {"enabled": False, "scale": 2.0}}}}
    out = apply_overrides(
        cfg,
        ["pipeline.stages.sr.enabled=true", "pipeline.stages.sr.scale=3"],
    )
    assert out["pipeline"]["stages"]["sr"]["enabled"] is True
    assert out["pipeline"]["stages"]["sr"]["scale"] == 3


def test_apply_overrides_string():
    cfg = {"pipeline": {"stages": {"pseco": {"prompt": "a"}}}}
    out = apply_overrides(cfg, ["pipeline.stages.pseco.prompt=bagged apple"])
    assert out["pipeline"]["stages"]["pseco"]["prompt"] == "bagged apple"


def test_apply_overrides_rejects_bad_path():
    with pytest.raises(KeyError):
        apply_overrides({"a": {"b": 1}}, ["a.c=2"])


def test_apply_overrides_rejects_missing_equal():
    with pytest.raises(ValueError):
        apply_overrides({"a": 1}, ["apple"])


def test_load_with_overrides(tmp_path):
    p = _write(
        tmp_path,
        """
        device: cpu
        pipeline:
          stages:
            pseco:
              enabled: true
              sam_checkpoint: a
              decoder_checkpoint: b
              mlp_checkpoint: c
              prompt: apple
        """,
    )
    cfg = load_config(p, overrides=["pipeline.stages.pseco.prompt=pear"])
    assert cfg.pipeline.stages.pseco.prompt == "pear"
