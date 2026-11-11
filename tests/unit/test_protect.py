import copy
import csv
import dataclasses
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

import counting.evaluation.record as record_module
import counting.evaluation.runner as runner_module
import counting.evaluation.sweep as sweep_module
import tests.integration.test_eval_synth_smoke as smoke
from counting.config.eval_schema import EvalAppConfig
from counting.config.loader import load_yaml_model
from counting.data.formats.orchard_synth import SynthRun
from counting.evaluation.cli import eval_app
from counting.evaluation.protect import (
    LOG_HEADER, ProtectedRequest, append_log_row, check_test_bundles, check_test_caches, check_test_grid,
    check_test_log, check_test_request, check_test_rungs, detect_log_row, earlier_log_runs,
)
from counting.evaluation.runner import CheckFailed, ProtectionError, load_ladder, rung_config
from counting.evaluation.sweep import Grid, load_grid
from tests.unit.cachefake import write_truth_cache
from tests.unit.conftest import budget_seconds
from tests.unit.synthfake import SIDECAR_COMMIT, fake_cfg, load_splits

ROOT = Path(__file__).resolve().parents[2]
SWEEPS = ROOT / "configs" / "eval" / "sweeps"
LADDER = load_ladder(SWEEPS / "ladder.yaml")
L5LADDER = load_ladder(SWEEPS / "ladder_l5.yaml")
FROZEN = "configs/eval/frozen/apple_test.yaml"
MISSING = "configs/eval/frozen/apple_missing.yaml"     # its bundle path does not exist
OLD_LOG = "configs/eval/frozen/apple_old_log.yaml"     # MISSING, with its own output folder `old_log`
CLEAN = {"commit": "a" * 40, "dirty": False, "diff_sha256": None}
runner = CliRunner()


def _git(repo, *args):
    base = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]
    return subprocess.run(base + list(args), check=True, capture_output=True).stdout.decode().strip()


def _commit(repo):
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "x")


def _make_frozen_repo(repo, out, configs, outs=None, patches=None, files=None):
    """A git repo with copies of the eval files a test run reads, and `configs` ({repo path: runs}) committed.

    Each config is `apple_dev.yaml` with split test, tuning false, stop_after y1 and output under `out`
    (or under `outs[repo path]`), with `patches[repo path]` ({section: {key: value}}) merged into its
    sections one level deep. `files` ({repo path: text}) are written as they are and committed too.
    """
    for rel in ("splits.yaml", "criteria.yaml", "sweeps/ladder.yaml", "sweeps/ladder_l5.yaml",
                "sweeps/pose_noise.yaml", "sweeps/det_noise.yaml"):
        (repo / "configs" / "eval" / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / "configs" / "eval" / rel, repo / "configs" / "eval" / rel)
    raw = yaml.safe_load((ROOT / "configs" / "eval" / "apple_dev.yaml").read_text(encoding="utf-8"))
    for rel, runs in configs.items():
        conf = {**raw, "run_name": "apple_test", "split": "test", "tuning": False, "stop_after": "y1",
                "runs": [str(r) for r in runs], "output_dir": str((outs or {}).get(rel, out))}
        for section, values in (patches or {}).get(rel, {}).items():
            conf[section] = {**(conf.get(section) or {}), **values}
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(yaml.safe_dump(conf), encoding="utf-8")
    for rel, text in (files or {}).items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text, encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    _commit(repo)
    return repo


@pytest.fixture(scope="module")
def frozen_repo(tmp_path_factory, heldout_run, apple_run):
    root = tmp_path_factory.mktemp("frozen")
    return _make_frozen_repo(root / "repo", root / "out", {
        FROZEN: [heldout_run], MISSING: [root / "missing"], "configs/eval/frozen/apple_seed5.yaml": [apple_run],
        "configs/eval/apple_loose.yaml": [root / "missing"], OLD_LOG: [root / "missing"]},
        outs={OLD_LOG: root / "old_log"})


def _check(repo, config=None, *, sets=(), runs=(), files=None, grid=None, ladder=None, code=CLEAN, overrides=None):
    """`check_test_request` on the frozen config's settings; `config` is the --config path (default: absolute)."""
    if files is None:
        files = {"--splits": repo / "configs/eval/splits.yaml", "--criteria": repo / "configs/eval/criteria.yaml"}
    return check_test_request(load_yaml_model(repo / FROZEN, EvalAppConfig, overrides=overrides),
                              config_path=repo / FROZEN if config is None else config, sets=sets, runs=runs,
                              files=files, grid=grid, ladder=ladder, repo=repo, code=code)


def _with_commit(run, commit):
    out = copy.copy(run)
    out.meta = dataclasses.replace(run.meta, synth_commit=commit)
    return out


def test_synth_smoke_reads_dev_bundles_only(heldout_run, tmp_path, monkeypatch):
    monkeypatch.setattr(smoke, "BASE", heldout_run.parent)
    with pytest.raises(pytest.fail.Exception, match="s1000: synth tests read dev bundles only; seed 1000 is test"):
        smoke._open(heldout_run.name)
    (tmp_path / "s7").mkdir()
    (tmp_path / "s7" / "manifest.json").write_text('{"args": {"seed": 7}}', encoding="utf-8")   # in no split
    monkeypatch.setattr(smoke, "BASE", tmp_path)
    with pytest.raises(pytest.fail.Exception, match="s7: synth tests read dev bundles only; seed 7 is none$"):
        smoke._open("s7")


def test_test_budget_must_be_a_positive_finite_number():
    assert budget_seconds(None) is None and budget_seconds("") is None and budget_seconds("15") == 15.0
    for raw in ("x", "nan", "inf", "-inf", "0", "-1"):
        with pytest.raises(pytest.UsageError, match=re.escape(f"must be a number of seconds, got {raw!r}")):
            budget_seconds(raw)


def test_test_request_passes_with_a_frozen_config_and_tracked_files(frozen_repo, monkeypatch):
    assert _check(frozen_repo) == ProtectedRequest(config_rel=FROZEN, code=CLEAN)
    monkeypatch.chdir(frozen_repo)
    sweep_files = {"--splits": "configs/eval/splits.yaml", "--criteria": "configs/eval/criteria.yaml",
                   "--ladder": "configs/eval/sweeps/ladder.yaml", "--grid": "configs/eval/sweeps/pose_noise.yaml"}
    got = _check(frozen_repo, config="configs/eval/frozen/../frozen/apple_test.yaml", files=sweep_files,
                 grid=load_grid("configs/eval/sweeps/pose_noise.yaml"), ladder=LADDER, code=None)
    assert got.config_rel == FROZEN and got.code == {"commit": _git(frozen_repo, "rev-parse", "HEAD"), "dirty": False,
                                                    "diff_sha256": None}
    with pytest.raises(ValueError, match="needs a split=test config"):
        check_test_request(load_yaml_model(ROOT / "configs/eval/apple_dev.yaml", EvalAppConfig), config_path=FROZEN,
                           sets=(), runs=(), files={}, grid=None, ladder=None, repo=frozen_repo, code=CLEAN)


def test_test_request_refusals_name_each_condition(frozen_repo, tmp_path):
    outside = tmp_path / "criteria.yaml"
    shutil.copy(frozen_repo / "configs/eval/criteria.yaml", outside)
    splits = frozen_repo / "configs/eval/splits.yaml"
    cases = [
        (dict(config=ROOT / "configs/eval/apple_dev.yaml"), "split=test needs a config file under configs/eval/frozen/"),
        (dict(config=frozen_repo / "configs/eval/apple_loose.yaml"),
         "split=test needs a config file under configs/eval/frozen/"),
        (dict(config=frozen_repo / "configs/eval/frozen/../apple_loose.yaml"),
         "split=test needs a config file under configs/eval/frozen/"),
        (dict(sets=("stop_after=y1",)), "split=test takes no --set or --run"),
        (dict(runs=("bundle",)), "split=test takes no --set or --run"),
        (dict(code={**CLEAN, "dirty": True, "diff_sha256": "d" * 64}), "split=test needs a clean, committed counting"),
        (dict(code={"commit": None, "dirty": False, "diff_sha256": None}), "split=test needs a clean, committed"),
        (dict(config=frozen_repo / "configs/eval/frozen/not_tracked.yaml"),
         "split=test needs --config to be a file tracked"),
        (dict(files={"--splits": splits, "--criteria": outside}),
         "split=test needs --criteria to be a file tracked in the counting repository"),
        (dict(files={"--splits": frozen_repo / "configs/eval/nope.yaml"}), "split=test needs --splits to be a file"),
        (dict(files={"--splits": splits, "--ladder": SWEEPS / "ladder.yaml"}), "split=test needs --ladder to be a file"),
        (dict(overrides=["detector.cache=/no/such/cache"]),
         "split=test with detector.cache needs detector.digests in the frozen config"),
    ]
    for kw, phrase in cases:
        with pytest.raises(ProtectionError, match=re.escape(phrase)):
            _check(frozen_repo, **kw)


def test_only_pose_noise_is_a_test_grid():
    first = {"dedup_modes.yaml": "point 'self_side' changes stages.dedup",
             "det_noise.yaml": "point 'detector.noise.miss_p=0.1' changes stages.det"}
    for path in sorted(SWEEPS.rglob("*.yaml")):
        name = str(path.relative_to(SWEEPS))
        if name in ("ladder.yaml", "ladder_l5.yaml"):
            continue
        if name == "pose_noise.yaml":
            check_test_grid(load_grid(path), L5LADDER)
            continue
        with pytest.raises(ProtectionError, match=re.escape(first.get(name, "split=test grid may change only"))):
            check_test_grid(load_grid(path), L5LADDER)


def test_test_grid_refuses_seeds_bare_sections_and_missing_bases():
    def grid(bases=("L3",), factors=None, combos=None):
        return Grid(path=Path("g.yaml"), bases=bases, factors=factors or {}, product={}, combos=combos or {},
                    common={})

    check_test_grid(grid(factors={"detector.noise.miss_p": [0.1]}), LADDER)
    cases = [(grid(factors={"noise.seeds": [[0, 1]]}), "changes noise.seeds"),
             (grid(combos={"bare": {"noise": {}}}), "point 'bare' changes noise"),
             (grid(combos={"det": {"detector": {"noise": {}}}}), "point 'det' changes detector.noise"),
             (grid(bases=(), factors={"noise.pass_bias_m": [0.02]}), "split=test grid needs bases, and only ladder rung"),
             (grid(bases=("L9",), factors={"noise.pass_bias_m": [0.02]}), "got ['L9']")]
    for g, phrase in cases:
        with pytest.raises(ProtectionError, match=re.escape(phrase)):
            check_test_grid(g, LADDER)
    with pytest.raises(ProtectionError, match="needs bases"):
        check_test_grid(grid(factors={"noise.pass_bias_m": [0.02]}), None)


def test_test_bundles_need_test_seeds_and_one_sidecar_commit(heldout_run, apple_run):
    run = SynthRun.open(heldout_run)
    assert run.meta.synth_commit == SIDECAR_COMMIT
    check_test_bundles([run, _with_commit(run, SIDECAR_COMMIT)], load_splits())
    cases = [([SynthRun.open(apple_run)], "split=test needs a test seed; seed 5 belongs to split dev"),
             ([_with_commit(run, None)], "s1000: split=test needs the orchard_synth.commit sidecar"),
             ([run, _with_commit(run, "d" * 40)],
              f"split=test needs one orchard_synth commit across the bundles, got ['{'c' * 40}', '{'d' * 40}']")]
    for bundles, phrase in cases:
        with pytest.raises(ProtectionError, match=re.escape(phrase)):
            check_test_bundles(bundles, load_splits())


def test_test_log_rows_and_earlier_runs(tmp_path):
    log = tmp_path / "out" / "test_log.csv"
    assert earlier_log_runs(log, config_hash="h", extra="e") == 0
    row = dict(zip(LOG_HEADER, ["2026-11-05T00:00:00+00:00", "id1", "ladder", FROZEN, "h", "ladder:x,y:True:criteria:c",
                                "f1;f2", "a" * 40, "c" * 64, "3/5"]))
    append_log_row(log, row)
    append_log_row(log, {**row, "experiment_id": "id2"})
    third = {**row, "experiment_id": "id3", "extra": "other"}
    append_log_row(log, {k: third[k] for k in reversed(LOG_HEADER)})       # any key order; written in LOG_HEADER order
    lines = log.read_text(encoding="utf-8").splitlines()
    assert lines[0] == ",".join(LOG_HEADER) and len(lines) == 4
    assert [r["experiment_id"] for r in csv.DictReader(lines)] == ["id1", "id2", "id3"]
    assert next(csv.DictReader(lines))["extra"] == "ladder:x,y:True:criteria:c"       # quoted, not split
    assert earlier_log_runs(log, config_hash="h", extra="ladder:x,y:True:criteria:c") == 2
    assert earlier_log_runs(log, config_hash="h2", extra="other") == 0
    with pytest.raises(ValueError, match="a test log row needs the keys"):
        append_log_row(log, {k: row[k] for k in LOG_HEADER[:-1]})
    check_test_log(log)
    check_test_log(tmp_path / "none.csv")
    (tmp_path / "empty.csv").write_text("", encoding="utf-8")
    check_test_log(tmp_path / "empty.csv")
    old = tmp_path / "old.csv"                       # an older column set: no config_hash or extra
    old.write_text("date,experiment_id,kind,config,verdict\n2026-01-01,id0,ladder,x,1/5\n", encoding="utf-8")
    phrase = f"{old}: a test log needs the columns {','.join(LOG_HEADER)} in this order, got date,experiment_id,"
    for call in (lambda: check_test_log(old), lambda: earlier_log_runs(old, config_hash="h", extra="e")):
        with pytest.raises(ValueError, match=re.escape(phrase)):
            call()


def test_test_caches_must_match_the_pins_and_the_detect_log(heldout_run, tmp_path):
    run = SynthRun.open(heldout_run)
    rid, fp = run.meta.run_id, run.meta.fingerprint
    cache_dir = tmp_path / "detect" / "0123456789abcdef"
    digest = write_truth_cache(heldout_run, cache_dir)
    base = ("split=test", "tuning=false", f"detector.cache={cache_dir}")

    def cfg(*sets):
        return fake_cfg(heldout_run, "apple", *base, *sets)

    def pin(d):
        return f"detector.digests={{{rid}: '{d}'}}"

    for c, phrase in ((cfg(f"detector.digests={{other: '{digest}'}}"), f"detector.digests pins no cache for run {rid}"),
                      (cfg(pin("0" * 64)), f"cache digest {digest[:16]} differs from the pinned {'0' * 16}"),
                      (cfg(pin(digest)), f"no counting detect row for cache {digest[:16]} of bundle {fp[:12]}")):
        with pytest.raises(ProtectionError, match=re.escape(phrase)):
            check_test_caches([run], c)
    log = cache_dir.parent / "test_log.csv"
    row = dict(date="d", config_rel="configs/detect/frozen/d.yaml", config_hash="h", weights_sha256="f" * 64,
               digests=[digest], commit="c")
    append_log_row(log, detect_log_row(fingerprints=["0" * 64], **row))           # another bundle's row
    with pytest.raises(ProtectionError, match="no counting detect row"):
        check_test_caches([run], cfg(pin(digest)))
    append_log_row(log, detect_log_row(fingerprints=[fp], **row))
    assert check_test_caches([run], cfg(pin(digest))) == [{"run_id": rid, "digest": digest}]
    gone = fake_cfg(heldout_run, "apple", *base[:2], f"detector.cache={tmp_path / 'none'}", pin(digest))
    with pytest.raises(ProtectionError, match="split=test needs the pinned detection cache file"):
        check_test_caches([run], gone)


def test_test_runs_take_cached_rung_settings_from_the_frozen_config(heldout_run, tmp_path):
    cfg = fake_cfg(heldout_run, "apple", "split=test", "tuning=false", f"detector.cache={tmp_path}",
                   f"detector.digests={{r: '{'a' * 64}'}}")
    ladder = {"rungs": {"L5": {"stages": {"det": "cached"}}, "L3h": {"tracker": {"score_hi": 0.7}},
                        "L5c": {"stages": {"det": "cached"}, "detector": {"cache": str(tmp_path / "x")}},
                        "L5h": {"stages": {"det": "cached"}, "tracker": {"score_hi": 0.7}}}}
    check_test_rungs(cfg, {n: rung_config(cfg, ladder, n) for n in ("L5", "L3h")})
    for name in ("L5c", "L5h"):
        with pytest.raises(ProtectionError, match=re.escape(f"split=test: rung {name} changes the detector or "
                                                            "tracker settings of a cached rung")):
            check_test_rungs(cfg, {name: rung_config(cfg, ladder, name)})


def _invoke(repo, monkeypatch, *args):
    monkeypatch.setattr(record_module, "counting_repo", lambda: repo)
    monkeypatch.chdir(repo)
    return runner.invoke(eval_app, list(args))


def test_eval_refuses_test_requests_before_scoring(frozen_repo, tmp_path, monkeypatch):
    out, missing = frozen_repo.parent / "out", frozen_repo.parent / "missing"
    copy_of = {}
    for rel in ("criteria.yaml", "splits.yaml", "sweeps/ladder.yaml", "sweeps/pose_noise.yaml"):
        copy_of[rel] = str(tmp_path / Path(rel).name)          # same bytes, outside the repository
        shutil.copy(frozen_repo / "configs/eval" / rel, copy_of[rel])
    cases = [
        (["ladder", "--config", "configs/eval/apple_loose.yaml", "--set", "split=test", "--set", "tuning=false"],
         "split=test needs a config file under"),
        (["run", "--config", MISSING, "--run", str(missing)], "split=test takes no --set or --run"),
        (["ladder", "--config", MISSING, "--set", "stop_after=y1"], "split=test takes no --set or --run"),
        (["ladder", "--config", MISSING, "--criteria", copy_of["criteria.yaml"]],
         "split=test needs --criteria to be a file"),
        (["ladder", "--config", MISSING, "--splits", copy_of["splits.yaml"]], "split=test needs --splits to be a file"),
        (["ladder", "--config", MISSING, "--ladder", copy_of["sweeps/ladder.yaml"]],
         "split=test needs --ladder to be a file"),
        (["sweep", "--config", MISSING, "--grid", copy_of["sweeps/pose_noise.yaml"]],
         "split=test needs --grid to be a file"),
        (["sweep", "--config", MISSING, "--grid", "configs/eval/sweeps/det_noise.yaml"],
         "point 'detector.noise.miss_p=0.1' changes stages.det"),
    ]
    for args, phrase in cases:     # a check that ran after opening the bundles would say "no manifest.json"
        r = _invoke(frozen_repo, monkeypatch, *args)
        assert r.exit_code == 1 and phrase in r.output and "manifest" not in r.output, (args, r.output)
    r = _invoke(frozen_repo, monkeypatch, "ladder", "--config", "configs/eval/frozen/apple_seed5.yaml")
    assert r.exit_code == 1 and "split=test needs a test seed; seed 5 belongs to split dev" in r.output, r.output
    r = _invoke(frozen_repo, monkeypatch, "ladder", "--config", MISSING, "--set", "tuning=true")
    assert r.exit_code == 2 and "split=test requires tuning=false" in r.output, r.output
    assert not out.exists()


def test_eval_checks_the_test_log_columns_before_opening_bundles(frozen_repo, monkeypatch):
    log = frozen_repo.parent / "old_log" / "test_log.csv"
    log.parent.mkdir()
    old = "date,experiment_id,kind,config,verdict\n2026-01-01,id0,ladder,x,1/5\n"
    log.write_text(old, encoding="utf-8")
    r = _invoke(frozen_repo, monkeypatch, "ladder", "--config", OLD_LOG)
    assert r.exit_code == 2 and "test_log.csv: a test log needs the columns" in r.output \
        and "manifest" not in r.output, r.output
    assert log.read_text(encoding="utf-8") == old and [p.name for p in log.parent.iterdir()] == ["test_log.csv"]


def test_eval_run_and_sweep_open_the_gate_only_after_the_checks(frozen_repo, monkeypatch):
    seen = []

    def stub(*args, allow_test=False, **kw):
        seen.append(allow_test)
        raise CheckFailed("scoring reached")

    monkeypatch.setattr(runner_module, "run_config", stub)
    monkeypatch.setattr(sweep_module, "run_sweep", stub)
    for args in (["run", "--config", FROZEN],
                 ["sweep", "--config", FROZEN, "--grid", "configs/eval/sweeps/pose_noise.yaml"]):
        r = _invoke(frozen_repo, monkeypatch, *args)
        assert r.exit_code == 1 and "error: scoring reached" in r.output, r.output
    assert seen == [True, True]


def test_eval_ladder_on_the_test_split_logs_every_run_and_warns_on_reruns(heldout_run, tmp_path, monkeypatch):
    out = tmp_path / "out"
    repo = _make_frozen_repo(tmp_path / "repo", out, {FROZEN: [heldout_run]})
    log = out / "test_log.csv"

    def ladder():
        r = _invoke(repo, monkeypatch, "ladder", "--config", FROZEN, "--rungs", "L1")
        assert r.exit_code == 0, r.output
        folder = Path(r.output.splitlines()[0])
        rows = list(csv.DictReader(log.read_text(encoding="utf-8").splitlines()))
        return r.output, folder, rows

    out1, d1, rows = ladder()
    res = json.loads((d1 / "results.json").read_text(encoding="utf-8"))
    rec = json.loads((d1 / "record.json").read_text(encoding="utf-8"))
    verdicts = res["verdicts"]["L1"]
    assert res["split"] == "test" and res["schema_version"] == 9 and "warning:" not in out1
    assert [v["metric"] for v in verdicts] == ["y1.e1_median", "y1.e1_max", "y1.dup_rate", "y1.miss_rate",
                                               "y1.merge_rate"] and res["pooled"]["L1"]["n_runs"] == 1
    assert "**판정 (test, 장면 시드 1개)**" in (d1 / "report.md").read_text(encoding="utf-8")
    (row,) = rows
    crit_sha = hashlib.sha256((repo / "configs/eval/criteria.yaml").read_bytes()).hexdigest()
    assert tuple(row) == LOG_HEADER and row["experiment_id"] == d1.name == res["experiment_id"]
    assert (row["kind"], row["config"], row["config_hash"]) == ("ladder", FROZEN, rec["config_hash"])
    assert row["fingerprints"] == ";".join(sorted(d["fingerprint"] for d in rec["data"]))
    assert row["commit"] == rec["code"]["commit"] == _git(repo, "rev-parse", "HEAD") and rec["code"]["dirty"] is False
    assert row["criteria_sha256"] == rec["extra"]["criteria_sha256"] == crit_sha
    assert row["extra"].endswith(f":criteria:{crit_sha[:16]}")
    assert row["verdict"] == f"{sum(v['ok'] is True for v in verdicts)}/5"
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d.*\+00:00", row["date"])

    (repo / "NOTE.txt").write_text("a code fix\n", encoding="utf-8")     # a new commit, same config and request
    _commit(repo)
    out2, d2, rows = ladder()
    assert d2 != d1 and len(rows) == 2 and rows[1]["experiment_id"] == d2.name
    assert f"warning: test_log.csv already lists 1 test run(s) of config {row['config_hash']}" in out2


def test_eval_ladder_on_test_with_L5_checks_the_caches_and_judges_detection(heldout_run, tmp_path, monkeypatch):
    run = SynthRun.open(heldout_run)
    rid, fp = run.meta.run_id, run.meta.fingerprint
    good = tmp_path / "detect" / "0123456789abcdef"
    digest = write_truth_cache(heldout_run, good)
    append_log_row(good.parent / "test_log.csv", detect_log_row(
        date="d", config_rel="configs/detect/frozen/d.yaml", config_hash="h", weights_sha256="f" * 64,
        fingerprints=[fp], digests=[digest], commit="c"))

    def det(cache_dir, digests=None):
        d = {"positives": "count1", "cache": str(cache_dir), "noise": None}
        return {"passes": {"step_m": 0.45}, "detector": d if digests is None else {**d, "digests": digests}}

    confs = {"configs/eval/frozen/l5.yaml": det(good, {rid: digest}), "configs/eval/frozen/l5_nopin.yaml": det(good),
             "configs/eval/frozen/l5_wrong.yaml": det(good, {rid: "0" * 64})}
    out = tmp_path / "out"
    patched = copy.deepcopy(dict(L5LADDER))
    patched["rungs"]["L5"] = {**patched["rungs"]["L5"], "tracker": {"score_hi": 0.7}}
    repo = _make_frozen_repo(tmp_path / "repo", out, {rel: [heldout_run] for rel in confs}, patches=confs,
                             files={"configs/eval/sweeps/ladder_l5h.yaml": yaml.safe_dump(patched)})
    l5 = ["ladder", "--ladder", "configs/eval/sweeps/ladder_l5.yaml", "--rungs", "L5", "--config"]
    l5h = ["ladder", "--ladder", "configs/eval/sweeps/ladder_l5h.yaml", "--rungs", "L5", "--config"]
    # The no-detect-log rejection is unit-tested directly (`test_test_caches_must_match_the_pins_and_the_detect_log`).
    for args, phrase in (([*l5, "configs/eval/frozen/l5_nopin.yaml"], "split=test with detector.cache needs detector.digests"),
                         ([*l5, "configs/eval/frozen/l5_wrong.yaml"], f"cache digest {digest[:16]} differs from the pinned"),
                         ([*l5h, "configs/eval/frozen/l5.yaml"],
                          "split=test: rung L5 changes the detector or tracker settings of a cached rung")):
        r = _invoke(repo, monkeypatch, *args)
        assert r.exit_code == 1 and phrase in r.output, (args, r.output)
    assert not out.exists()
    r = _invoke(repo, monkeypatch, *l5, "configs/eval/frozen/l5.yaml")
    assert r.exit_code == 0, r.output
    folder = Path(r.output.splitlines()[0])
    res = json.loads((folder / "results.json").read_text(encoding="utf-8"))
    (row,) = list(csv.DictReader((out / "test_log.csv").read_text(encoding="utf-8").splitlines()))
    rows = [v for vs in res["verdicts"].values() for v in vs]
    assert [v["metric"] for v in res["verdicts"]["L5"]] == ["y1.e1_median"] and len(res["verdicts"]["detection"]) == 4
    assert row["verdict"] == f"{sum(v['ok'] is True for v in rows)}/5" and digest in row["fingerprints"].split(";")
    report = (folder / "report.md").read_text(encoding="utf-8")
    assert "**판정 (test, 장면 시드 1개)**" in report and "| L5 검출 30° | detection.recall |" in report
    assert "budget" not in res and not (folder / "budget.csv").exists() and "오차 예산" not in report   # one rung
