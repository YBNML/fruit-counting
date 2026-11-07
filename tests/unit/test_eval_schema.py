import re
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from counting.config.eval_schema import (
    CachePostConfig, DetNoise, EvalAppConfig, LiftConfig, NoiseConfig, SplitsConfig, TrackerConfig,
)
from counting.config.hashing import config_hash
from counting.config.loader import load_yaml_model

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "configs" / "eval"


def _raw(**over):
    raw = yaml.safe_load((EVAL / "apple_dev.yaml").read_text(encoding="utf-8"))
    raw.update(over)
    return raw


def test_apple_dev_config_values():
    cfg = load_yaml_model(EVAL / "apple_dev.yaml", EvalAppConfig)
    assert (cfg.run_name, cfg.crop, cfg.split, cfg.tuning) == ("apple_dev", "apple", "dev", True)
    assert cfg.runs == ["~/YBNML_repo/orchard_synth/out/apple/robot3"]
    assert cfg.stages.model_dump() == {"det": "oracle", "track": "landmark", "center": "measured",
                                       "dedup": "merge", "assign": "grid"}
    assert cfg.lift.depth == "box_median" and cfg.lift.box_frac == 0.5
    assert cfg.lift.center_offset_m == {"fruit": 0.02, "bag": 0.03}
    t = cfg.tracker
    assert (t.gate_px_frac, t.gate_m, t.max_gap, t.repair_m) == (0.5, 0.05, None, 0.04)
    assert (t.min_obs, t.score_hi, t.score_lo) == (1, 0.5, 0.1)
    assert (cfg.dedup.radius_m, cfg.dedup.band_m) == (0.04, 0.10)
    assert cfg.noise.is_zero() and cfg.stop_after == "y6" and cfg.output_dir == "runs/eval"


def test_pear_dev_config_values():
    cfg = load_yaml_model(EVAL / "pear_dev.yaml", EvalAppConfig)
    assert cfg.crop == "pear"
    assert cfg.runs == ["~/YBNML_repo/orchard_synth/out/pear/robot3",
                        "~/YBNML_repo/orchard_synth/out/pear/robot3_occl"]
    assert cfg.stages.dedup == "self_side" and cfg.lift.center_offset_m == {"bag": 0.04}
    assert (cfg.tracker.gate_m, cfg.tracker.repair_m, cfg.dedup.radius_m) == (0.10, 0.09, 0.09)



# Values chosen on the val bundle; val and exp configs carry them, dev configs keep the defaults.
TUNED_ON_VAL = {
    "apple": {"stages": {"dedup": "hybrid"},
              "lift": {"depth": "box_far"},
              "tracker": {"depth_gate": "soft", "repair_m": 0.08},
              "dedup": {"radius_m": 0.09, "band_m": 0.05}},
    "pear": {"lift": {"depth": "box_far"},
             "tracker": {"depth_gate": "soft", "repair_m": 0.18},
             "dedup": {"radius_m": 0.135}},
}


@pytest.mark.parametrize("crop", ["apple", "pear"])
def test_val_and_exp_configs_follow_dev(crop):
    dev = load_yaml_model(EVAL / f"{crop}_dev.yaml", EvalAppConfig)
    splits = load_yaml_model(EVAL / "splits.yaml", SplitsConfig)
    for split in ("val", "exp"):
        cfg = load_yaml_model(EVAL / f"{crop}_{split}.yaml", EvalAppConfig)
        (seed,) = getattr(splits, split)
        assert cfg.runs == [f"~/datasets/orchard_synth/{crop}/{split}_s{seed}"]
        want = {**dev.model_dump(), "run_name": f"{crop}_{split}", "split": split, "runs": cfg.runs,
                "noise": {**dev.noise.model_dump(), "seeds": [0, 1, 2]}}
        for section, values in TUNED_ON_VAL[crop].items():
            want[section] = {**want[section], **values}
        assert cfg.model_dump() == want


@pytest.mark.parametrize("crop", ["apple", "pear"])
def test_frozen_test_config_follows_val(crop):
    val = load_yaml_model(EVAL / f"{crop}_val.yaml", EvalAppConfig)
    splits = load_yaml_model(EVAL / "splits.yaml", SplitsConfig)
    cfg = load_yaml_model(EVAL / "frozen" / f"{crop}_test_oracle.yaml", EvalAppConfig)
    assert cfg.runs == [f"~/datasets/orchard_synth/{crop}/test_s{seed}" for seed in splits.test]
    assert cfg.model_dump() == {**val.model_dump(), "run_name": f"{crop}_test_oracle", "split": "test",
                                "tuning": False, "runs": cfg.runs}


@pytest.mark.parametrize("crop", ["apple", "pear"])
def test_frozen_d1_configs_add_only_the_detector_settings(crop):
    oracle = load_yaml_model(EVAL / "frozen" / f"{crop}_test_oracle.yaml", EvalAppConfig).model_dump()
    d1 = load_yaml_model(EVAL / "frozen" / f"{crop}_test_d1.yaml", EvalAppConfig).model_dump()

    def strip(d):
        d = {k: v for k, v in d.items() if k not in ("run_name", "verdict_subsets")}
        d["detector"] = {k: v for k, v in d["detector"].items() if k not in ("cache", "post", "digests")}
        d["tracker"] = {k: v for k, v in d["tracker"].items() if k not in ("score_hi", "score_lo", "min_obs_cached")}
        return d

    assert strip(d1) == strip(oracle) and d1["run_name"] == f"{crop}_test_d1"
    assert d1["detector"]["cache"] == "runs/detect/88e8c93628216703"
    digests = d1["detector"]["digests"]
    assert len(digests) == 3 and all(k.startswith(f"{crop}_s100") for k in digests)
    assert all(re.fullmatch(r"[0-9a-f]{64}", v) for v in digests.values())


def test_unknown_key_rejected():
    raw = _raw()
    raw["stages"]["bogus"] = 1
    with pytest.raises(ValidationError):
        EvalAppConfig.model_validate(raw)


def test_cached_detector_rejects_oracle_track():
    raw = _raw()
    raw["stages"].update(det="cached", track="oracle")
    with pytest.raises(ValidationError, match="track=oracle"):
        EvalAppConfig.model_validate(raw)


def test_cached_detector_rejects_det_depth():
    raw = _raw()
    raw["stages"]["det"] = "cached"
    raw["lift"]["depth"] = "det_depth"
    with pytest.raises(ValidationError, match="det_depth"):
        EvalAppConfig.model_validate(raw)


def test_test_split_requires_tuning_false():
    with pytest.raises(ValidationError, match="tuning=false"):
        EvalAppConfig.model_validate(_raw(split="test", tuning=True))


def test_score_lo_above_hi_rejected():
    raw = _raw()
    raw["tracker"].update(score_lo=0.6, score_hi=0.5)
    with pytest.raises(ValidationError, match="score_lo"):
        EvalAppConfig.model_validate(raw)


def test_lift_offset_by_label():
    lift = LiftConfig(center_offset_m={"bag": 0.04})
    assert lift.offset(1) == 0.04 and lift.offset(0) == 0.0


def test_lift_depth_options():
    lift = LiftConfig()
    assert (lift.depth, lift.box_frac, lift.far_scale, lift.far_gap_m, lift.far_min_frac) == \
        ("box_median", 0.5, 1.0, 0.03, 0.25)
    assert LiftConfig(depth="box_far", far_min_frac=1.0).depth == "box_far"
    for bad in ({"far_scale": 0}, {"far_scale": 1.5}, {"far_gap_m": 0}, {"far_min_frac": 0},
                {"far_min_frac": 1.5}, {"depth": "box_near"}):
        with pytest.raises(ValidationError):
            LiftConfig(**bad)


def test_tracker_depth_gate_options():
    t = TrackerConfig()
    assert (t.depth_gate, t.soft_cost) == ("hard", 0.5)
    for bad in ({"depth_gate": "off"}, {"soft_cost": -0.1}):
        with pytest.raises(ValidationError):
            TrackerConfig(**bad)


def test_splits_file():
    s = load_yaml_model(EVAL / "splits.yaml", SplitsConfig)
    assert (s.dev, s.val, s.test, s.exp) == ([5], [200], [1000, 1001, 1002], [300])
    assert s.train == list(range(100, 120))
    assert s.split_of(1001) == "test" and s.split_of(5) == "dev" and s.split_of(7) is None


def test_splits_overlap_rejected():
    with pytest.raises(ValidationError, match="splits overlap"):
        SplitsConfig(dev=[5], val=[5], test=[1000], exp=[300], train=[100])


def test_criteria_file_values():
    c = yaml.safe_load((EVAL / "criteria.yaml").read_text(encoding="utf-8"))
    assert c["y1"]["L1"]["pear"]["e1_median"] == {"le": 0.02}
    assert c["y1"]["L1"]["pear"]["e1_max"] == {"le": 0.05}
    assert c["y1"]["L1"]["apple"]["e1_max"] == {"le": 0.05}
    assert c["y1"]["L3"]["apple"]["fp_off_rate"] == {"le": 0.02}
    assert c["y6"]["L3"]["pear"]["pos_err_p90_m"] == {"le": 0.05}
    assert c["y6"]["L3"]["apple"]["row_abs_e_pipe"] == {"le": 0.05}
    assert c["y6"]["L4_base"]["apple"]["row_abs_e_pipe"] == {"le": 0.10}
    assert c["y6"]["L5"]["apple"]["tree_mae_over_mean"] == {"le": 0.15}
    assert c["detection"]["L5"]["apple"]["recall"] == {"ge": 0.85}


def test_load_yaml_model_expands_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "s.yaml").write_text("dev: [5]\nval: [200]\ntest: [1000]\nexp: [300]\ntrain: [100]\n",
                                     encoding="utf-8")
    assert load_yaml_model("~/s.yaml", SplitsConfig).dev == [5]


def test_load_yaml_model_overrides():
    cfg = load_yaml_model(EVAL / "apple_dev.yaml", EvalAppConfig,
                          overrides=["tracker.max_gap=7", "stages.track=last_box"])
    assert cfg.tracker.max_gap == 7 and cfg.stages.track == "last_box"


def test_config_hash_generic_and_ignores_runtime_keys():
    a = EvalAppConfig.model_validate(_raw())
    b = EvalAppConfig.model_validate(_raw(runs=["elsewhere"], output_dir="x"))
    c = EvalAppConfig.model_validate(_raw(seed=1))
    assert len(config_hash(a)) == 16
    assert config_hash(a) == config_hash(b) != config_hash(c)


COMMITTED_HASHES = {"apple_dev": "38b74b45e42d9213", "pear_dev": "b85d292df18d8261", "apple_val": "b26494b8830a44fa",
                    "apple_exp": "7f6be6772929d5ed", "pear_val": "a16d22fe031ee666", "pear_exp": "8c67f0e894d833c7",
                    "frozen/apple_test_oracle": "a5d60d44aaf799f2",
                    "frozen/pear_test_oracle": "7fcb765786bad4bb",
                    "frozen/apple_test_d1": "a78a8e26beafabb6", "frozen/pear_test_d1": "1c6708892241aec8"}


def test_committed_config_hashes_unchanged():
    for name, h in COMMITTED_HASHES.items():
        assert config_hash(load_yaml_model(EVAL / f"{name}.yaml", EvalAppConfig)) == h, name


def test_det_noise_defaults_and_bounds():
    assert DetNoise().model_dump() == {"miss_p": 0.0, "fp_lambda": 0.0, "jitter_frac": 0.0, "split_p": 0.0,
                                       "split_visib_max": 0.6, "score": "fixed", "score_min": 0.3}
    for bad in ({"miss_p": 1.5}, {"miss_p": -0.1}, {"fp_lambda": -1}, {"fp_lambda": float("inf")},
                {"fp_lambda": "inf"}, {"jitter_frac": -0.1}, {"jitter_frac": float("nan")}, {"split_p": 2},
                {"split_visib_max": 0}, {"score": "beta"}, {"score_min": 1.5}, {"sigma": 0.1}):
        with pytest.raises(ValidationError):
            DetNoise(**bad)


def test_noise_factors_must_be_finite():
    for key in ("pose_trans_sigma_m", "pose_rot_sigma_deg", "pass_bias_m", "depth_rel_sigma"):
        for bad in (float("inf"), float("nan")):
            with pytest.raises(ValidationError):
                NoiseConfig(**{key: bad})
    with pytest.raises(ValidationError):
        load_yaml_model(EVAL / "apple_dev.yaml", EvalAppConfig,
                        overrides=["stages.det=oracle_noisy", "detector.noise.fp_lambda=inf"])


def test_det_noise_and_oracle_noisy_need_each_other():
    raw = _raw()
    raw["stages"]["det"] = "oracle_noisy"
    with pytest.raises(ValidationError, match="stages.det=oracle_noisy needs a detector.noise section"):
        EvalAppConfig.model_validate(raw)
    raw = _raw()
    raw["detector"]["noise"] = {"miss_p": 0.1}
    with pytest.raises(ValidationError, match="detector.noise needs stages.det=oracle_noisy, got stages.det=oracle"):
        EvalAppConfig.model_validate(raw)
    raw["stages"]["det"] = "oracle_noisy"
    assert EvalAppConfig.model_validate(raw).detector.noise == DetNoise(miss_p=0.1)
    zero = load_yaml_model(EVAL / "apple_dev.yaml", EvalAppConfig,
                           overrides=["stages.det=oracle_noisy", "detector.noise={}"])
    assert zero.detector.noise == DetNoise()


def test_noise_seeds_need_distinct_non_negative_values():
    for seeds, phrase in (([], "at least 1"), ([0, 0], "noise.seeds repeats a seed"),
                          ([-1], "greater than or equal to 0")):
        raw = _raw()
        raw["noise"]["seeds"] = seeds
        with pytest.raises(ValidationError, match=re.escape(phrase)):
            EvalAppConfig.model_validate(raw)


def test_cache_settings_stay_out_of_unchanged_configs():
    cfg = EvalAppConfig.model_validate(_raw())
    dump = cfg.model_dump(mode="json")
    assert set(dump["detector"]) == {"positives", "cache", "noise"} and "min_obs_cached" not in dump["tracker"]
    explicit = _raw(detector={"positives": "count1", "cache": None, "noise": None, "post": None, "digests": None})
    assert config_hash(EvalAppConfig.model_validate(explicit)) == config_hash(cfg)
    tuned = EvalAppConfig.model_validate(_raw(tracker={**_raw()["tracker"], "min_obs_cached": 2}))
    assert config_hash(tuned) != config_hash(cfg) and EvalAppConfig.model_validate(tuned.model_dump()) == tuned
    p = cfg.detector.post_settings()
    assert p == CachePostConfig() and (p.nms_iou, p.edge_snap_px, p.min_short_px) == (0.5, 2.0, 0.0)
    assert (tuned.tracker.min_obs_for("cached"), tuned.tracker.min_obs_for("oracle"),
            cfg.tracker.min_obs_for("cached")) == (2, 1, 1)


def test_cached_detector_settings_are_checked():
    det = {"positives": "count1", "noise": None}
    with pytest.raises(ValidationError, match=re.escape("stages.det=cached needs detector.cache")):
        EvalAppConfig.model_validate(_raw(stages={**_raw()["stages"], "det": "cached"}, detector={**det, "cache": None}))
    with pytest.raises(ValidationError, match="detector.digests needs detector.cache"):
        EvalAppConfig.model_validate(_raw(detector={**det, "cache": None, "digests": {"r": "a" * 64}}))
    with pytest.raises(ValidationError, match=re.escape("detector.digests[r] must be a 64-character lowercase hex")):
        EvalAppConfig.model_validate(_raw(detector={**det, "cache": "c", "digests": {"r": "A" * 64}}))
    for post in ({"nms_iou": 0}, {"edge_snap_px": -1}, {"min_short_px": -1}, {"nms": 0.5}):
        with pytest.raises(ValidationError):
            EvalAppConfig.model_validate(_raw(detector={**det, "cache": "c", "post": post}))
    ok = EvalAppConfig.model_validate(_raw(stages={**_raw()["stages"], "det": "cached"},
                                           detector={**det, "cache": "runs/detect/x", "digests": {"r": "a" * 64}}))
    assert ok.detector.cache == "runs/detect/x" and ok.detector.digests == {"r": "a" * 64}


def test_verdict_subsets_name_at_least_one_subset():
    with pytest.raises(ValidationError, match="verdict_subsets\n  List should have at least 1 item"):
        EvalAppConfig.model_validate(_raw(verdict_subsets=[]))
    assert EvalAppConfig.model_validate(_raw(verdict_subsets=["drop 20"])).verdict_subsets == ["drop 20"]
    assert "verdict_subsets" not in EvalAppConfig.model_validate(_raw()).model_dump()


def test_is_noisy():
    base = load_yaml_model(EVAL / "apple_val.yaml", EvalAppConfig)
    assert not base.is_noisy() and base.noise.seeds == [0, 1, 2]          # seeds alone are not noise
    for sets in (["noise.depth_dropout=0.05"], ["stages.det=oracle_noisy", "detector.noise={}"]):
        assert load_yaml_model(EVAL / "apple_val.yaml", EvalAppConfig, overrides=sets).is_noisy(), sets
