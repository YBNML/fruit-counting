from counting.config.hashing import config_hash
from counting.config.schema import AppConfig


def _cfg(**overrides):
    data = {
        "device": "cpu",
        "seed": 42,
        "output_dir": "./runs",
        "pipeline": {
            "stages": {
                "pseco": {
                    "enabled": True,
                    "sam_checkpoint": "a",
                    "decoder_checkpoint": "b",
                    "mlp_checkpoint": "c",
                }
            }
        },
    }
    data.update(overrides)
    return AppConfig.model_validate(data)


def test_hash_is_deterministic():
    a = config_hash(_cfg())
    b = config_hash(_cfg())
    assert a == b
    assert len(a) == 16


def test_hash_changes_with_relevant_field():
    a = config_hash(_cfg())
    b = config_hash(_cfg(seed=99))
    assert a != b


def test_hash_ignores_device_and_output_dir():
    """Device/output_dir are runtime concerns; they must not change the hash."""
    base = config_hash(_cfg())
    assert config_hash(_cfg(device="auto")) == base
    assert config_hash(_cfg(output_dir="./other")) == base


def test_hash_changes_with_checkpoint_path():
    base = config_hash(_cfg())

    # Changing sam_checkpoint should change the hash
    other = _cfg(
        pipeline={
            "stages": {
                "pseco": {
                    "enabled": True,
                    "sam_checkpoint": "DIFFERENT.pth",
                    "decoder_checkpoint": "b",
                    "mlp_checkpoint": "c",
                }
            }
        }
    )
    assert config_hash(other) != base
