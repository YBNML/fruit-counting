"""Deterministic results serialization, the experiment folder writer and the markdown
report. `render_report` and `criteria_rows` skip the Y6 sections
of a rung whose `stop_after=y1` left `rung["y6"]` as `None`.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np
import yaml

from counting.config.eval_schema import EvalAppConfig

SCHEMA_VERSION = 1


class ReproducibilityError(RuntimeError):
    """A rerun of an existing experiment id produced a different `results.json`."""


# --- determinism -----------------------------------------------------------------------


def round_floats(obj, ndigits: int = 6):
    """Recursively round floats to `ndigits`, turn NaN/inf into `None`, and convert numpy
    and tuple values to plain JSON-shaped Python values. Python `bool` passes through.
    """
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (float, np.floating)):
        x = float(obj)
        return None if math.isnan(x) or math.isinf(x) else round(x, ndigits)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return round_floats(obj.tolist(), ndigits)
    if isinstance(obj, (list, tuple)):
        return [round_floats(v, ndigits) for v in obj]
    if isinstance(obj, dict):
        return {k: round_floats(v, ndigits) for k, v in obj.items()}
    return obj


def results_bytes(results: dict) -> bytes:
    """Canonical JSON bytes of `results`: sorted keys, 6-digit floats, `null` for NaN."""
    return json.dumps(round_floats(results), sort_keys=True, indent=2, ensure_ascii=False,
                      allow_nan=False).encode("utf-8") + b"\n"


# --- experiment folder -------------------------------------------------------------------

_PER_PASS_HEADER = ["run_id", "rung", "pitch", "side", "aisle", "n_frames", "n_tracks", "n_no3d",
                    "U1", "G1", "e1", "fp_bg", "fp_off", "boundary", "dup_track", "dup_cross",
                    "miss_det", "miss_drop", "miss_merge", "impure"]
_PER_TREE_HEADER = ["run_id", "rung", "tree", "U", "U_truth_attr", "G6", "G6_count1", "G6_any",
                    "G6_strict", "G6_2", "e_pipe", "e_cover", "e_total", "mis_in", "mis_out"]


def _per_pass_rows(results: dict) -> list[list]:
    rows = []
    for run in results["runs"]:
        run_id = run["meta"]["run_id"]
        for rung_name, rung in run["rungs"].items():
            for p in rung["y1"]["passes"]:
                L = p["ledger"]
                rows.append([run_id, rung_name, p["pitch"], p["side"], p["aisle"], p["n_frames"],
                            p["n_tracks"], p["n_no3d"], p["U1"], p["G1"], p["e1"], L["fp_bg"],
                            L["fp_off"], L["boundary"], L["dup_track"], L["dup_cross"], L["miss_det"],
                            L["miss_drop"], L["miss_merge"], L["impure"]])
    return rows


def _per_tree_rows(results: dict) -> list[list]:
    rows = []
    for run in results["runs"]:
        run_id = run["meta"]["run_id"]
        for rung_name, rung in run["rungs"].items():
            if rung["y6"] is None:   # stop_after=y1 has no tree table to write
                continue
            for t in rung["y6"]["trees"]:
                rows.append([run_id, rung_name, t["tree"], t["U"], t["U_truth_attr"], t["G6"],
                            t["G6_count1"], t["G6_any"], t["G6_strict"], t["G6_2"], t["e_pipe"],
                            t["e_cover"], t["e_total"], t["mis_in"], t["mis_out"]])
    return rows


def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(round_floats(rows))


def write_experiment(
    out_root: Path, *, cfg: EvalAppConfig, results: dict, record: dict, criteria: Mapping | None,
) -> tuple[Path, bool]:
    """Write the experiment folder `out_root/<run_name>/<experiment_id>/`.

    Computes the new `results.json` bytes first; if a different `results.json` already
    exists there, raises `ReproducibilityError` before writing anything. Otherwise
    (re-)writes every file and returns `(folder, reproduced)`.
    """
    folder = Path(out_root) / results["run_name"] / results["experiment_id"]
    new_bytes = results_bytes(results)
    results_path = folder / "results.json"
    reproduced = False
    if results_path.exists():
        if results_path.read_bytes() != new_bytes:
            raise ReproducibilityError(str(folder))
        reproduced = True

    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.yaml").write_text(
        yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=True, allow_unicode=True), encoding="utf-8")
    (folder / "record.json").write_bytes(results_bytes(record))
    results_path.write_bytes(new_bytes)
    _write_csv(folder / "per_pass.csv", _PER_PASS_HEADER, _per_pass_rows(results))
    _write_csv(folder / "per_tree.csv", _PER_TREE_HEADER, _per_tree_rows(results))
    (folder / "report.md").write_text(render_report(results, criteria=criteria), encoding="utf-8")
    return folder, reproduced


# --- markdown report ----------------------------------------------------------------------


def _g(x) -> str:
    return format(x, "g")


def _pct(x) -> str:
    return "-" if x is None else format(x, "+.1%")


def _m(x) -> str:
    return "-" if x is None else f"{x:.3f}"


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines


_Y1_HEADERS = ["기울기", "면", "통로", "프레임", "U1", "G1", "e1", "Dup", "Miss", "FP_off", "3D 없음"]
_Y6_TREE_HEADERS = ["그루", "U", "U_정답귀속", "G6", "G6①", "G6any", "G6strict", "G6②",
                    "e_pipe", "e_cover", "e_total", "유입", "유출"]
_LEDGER_FIELDS = ["n_pred", "n_truth", "fp_bg", "fp_off", "boundary", "dup_track", "dup_cross",
                  "miss_det", "miss_drop", "miss_merge", "impure"]
_CONTRIB_LABELS = {
    ("L0", "L1"): "L0→L1 (추적)",
    ("L1", "L2a"): "L1→L2a (중복 제거)",
    ("L2a", "L2b"): "L2a→L2b (역투영·보정)",
    ("L2b", "L2c"): "L2b→L2c (상자 깊이)",
    ("L2c", "L3"): "L2c→L3 (줄 거르기·귀속)",
}
_OP_SYMBOL = {"le": "≤", "ge": "≥"}
# Criteria metrics that are ratios, shown in percent like everywhere else in the report.
_RATIO_METRICS = frozenset({
    "e1_median", "e1_max", "dup_rate", "miss_rate", "merge_rate", "fp_off_rate", "row_abs_e_pipe",
    "tree_mae_over_mean", "attribution_error", "excess_rate", "recall", "precision",
})


def _y1_rows(passes: list[dict]) -> list[list[str]]:
    rows = []
    for p in passes:
        L = p["ledger"]
        dup = L["dup_track"] + L["dup_cross"]
        miss = L["miss_det"] + L["miss_drop"] + L["miss_merge"]
        rows.append([_g(p["pitch"]), p["side"], str(p["aisle"]), str(p["n_frames"]), str(p["U1"]),
                    str(p["G1"]), _pct(p["e1"]), str(dup), str(miss), str(L["fp_off"]), str(p["n_no3d"])])
    return rows


def _y1_summary_line(summary: dict) -> str:
    return (f"- Y1 요약: e1 중앙값 {_pct(summary['e1_median'])}, e1 최대 {_pct(summary['e1_max'])}, "
           f"Dup율 {_pct(summary['dup_rate'])}, Miss율 {_pct(summary['miss_rate'])}, "
           f"Merge율 {_pct(summary['merge_rate'])}, FP_off율 {_pct(summary['fp_off_rate'])}, "
           f"naive U {summary['naive_U']}")


def _y6_tree_rows(trees: list[dict]) -> list[list[str]]:
    rows = []
    for t in trees:
        rows.append([str(t["tree"]), str(t["U"]), str(t["U_truth_attr"]), str(t["G6"]), str(t["G6_count1"]),
                    str(t["G6_any"]), str(t["G6_strict"]), str(t["G6_2"]), _pct(t["e_pipe"]),
                    _pct(t["e_cover"]), _pct(t["e_total"]), str(t["mis_in"]), str(t["mis_out"])])
    return rows


def _mult(x) -> str:
    return "-" if x is None else f"{x:.2f}배"


def _row_sum_line(row: dict) -> str:
    return (f"- 줄 합: U {row['U']}, G6 {row['G6']}, G6② {row['G6_2']}, e_pipe {_pct(row['e_pipe'])}, "
           f"e_cover {_pct(row['e_cover'])}, e_total {_pct(row['e_total'])}, "
           f"naive {row['naive_U']} ({_mult(row['naive_ratio'])})")


def _ledger_table(ledger_g1: dict, ledger_any: dict) -> list[str]:
    headers = ["판", *_LEDGER_FIELDS]
    rows = [["G①", *[str(ledger_g1[f]) for f in _LEDGER_FIELDS]],
            ["any", *[str(ledger_any[f]) for f in _LEDGER_FIELDS]]]
    return _table(headers, rows)


def _metrics_line(metrics: dict) -> str:
    return (f"- 지표: row_abs_e_pipe {_pct(metrics['row_abs_e_pipe'])}, "
           f"tree_mae_over_mean {_pct(metrics['tree_mae_over_mean'])}, "
           f"attribution_error {_pct(metrics['attribution_error'])}, "
           f"pos_err_p50_m {_m(metrics['pos_err_p50_m'])}, pos_err_p90_m {_m(metrics['pos_err_p90_m'])}, "
           f"miss_rate {_pct(metrics['miss_rate'])}, excess_rate {_pct(metrics['excess_rate'])}")


def _contrib_label(a: str, b: str) -> str:
    return _CONTRIB_LABELS.get((a, b), f"{a}→{b}")


def _delta(a, b):
    return None if a is None or b is None else b - a


def _y6_delta(ra: dict, rb: dict, key_path: tuple[str, ...]):
    if ra["y6"] is None or rb["y6"] is None:
        return None
    a, b = ra["y6"], rb["y6"]
    for k in key_path:
        a, b = a[k], b[k]
    return _delta(a, b)


def _contrib_table(chain: list[str], rungs: dict) -> list[str]:
    headers = ["구간", "Δ Y1 e1 중앙값", "Δ 줄 e_pipe", "Δ 그루 MAE÷평균"]
    rows = []
    for a, b in zip(chain, chain[1:]):
        ra, rb = rungs[a], rungs[b]
        d_e1 = _delta(ra["y1"]["summary"]["e1_median"], rb["y1"]["summary"]["e1_median"])
        d_epipe = _y6_delta(ra, rb, ("row", "e_pipe"))
        d_mae = _y6_delta(ra, rb, ("metrics", "tree_mae_over_mean"))
        rows.append([_contrib_label(a, b), _pct(d_e1), _pct(d_epipe), _pct(d_mae)])
    return _table(headers, rows)


def _cell_table(cells: list[dict]) -> list[str]:
    headers = ["추적", "중복 제거", "e_pipe", "dup_track", "dup_cross"]
    rows = [[c["track"], c["dedup"], _pct(c["e_pipe"]),
            "-" if c["dup_track"] is None else str(c["dup_track"]),
            "-" if c["dup_cross"] is None else str(c["dup_cross"])] for c in cells]
    return _table(headers, rows)


def criteria_rows(rung: dict, criteria: Mapping, rung_name: str, crop: str) -> list[dict]:
    """Compare `rung`'s Y1/Y6 metrics to `criteria[kind][rung_name][crop]` (`kind` is `y1`/`y6`).

    A metric row's `ok` is `None` when the metric is absent from `rung`. The whole Y6 half
    is skipped when `rung["y6"] is None`; a `rung_name`/`crop` missing from
    `criteria` contributes no rows for that half.
    """
    rows = []
    for kind, metrics_key in (("y1", "summary"), ("y6", "metrics")):
        if kind == "y6" and rung.get("y6") is None:
            continue
        spec = ((criteria.get(kind) or {}).get(rung_name) or {}).get(crop)
        if not spec:
            continue
        values = rung[kind][metrics_key]
        for key, lim in spec.items():
            (op, limit), = lim.items()
            value = values.get(key)
            if value is None:
                ok = None
            elif op == "le":
                ok = value <= limit
            elif op == "ge":
                ok = value >= limit
            else:
                raise ValueError(f"unknown criteria op: {op!r}")
            rows.append({"metric": f"{kind}.{key}", "value": value, "op": op, "limit": limit, "ok": ok})
    return rows


def _criteria_cell(metric: str, x) -> str:
    """A criteria value or limit: percent for a ratio, metres (`*_m`) to 3 decimals, else as is."""
    key = metric.partition(".")[2]
    if key in _RATIO_METRICS:
        return _pct(x)
    if key.endswith("_m"):
        return _m(x)
    return "-" if x is None else _g(x)


def _criteria_table(entries: list[tuple[str, list[dict]]]) -> list[str]:
    """One table for every rung that has criteria rows; the first column names the rung."""
    headers = ["칸", "지표", "값", "기준", "판정"]
    out = []
    for rung_name, rows in entries:
        for r in rows:
            value = _criteria_cell(r["metric"], r["value"])
            limit = f"{_OP_SYMBOL[r['op']]}{_criteria_cell(r['metric'], r['limit'])}"
            verdict = "-" if r["ok"] is None else ("OK" if r["ok"] else "FAIL")
            out.append([rung_name, r["metric"], value, limit, verdict])
    return _table(headers, out)


def render_report(results: dict, *, criteria: Mapping | None) -> str:
    """Build the human-readable `report.md` for one experiment."""
    lines = [f"# Y1·Y6 합성 채점 — {results['run_name']} ({results['experiment_id']})",
             "> 합성 렌더 · 정답 자세 · 보정 전(U). 실제 정확도가 아니다."]
    if results["split"] == "dev":
        lines.append("> dev 묶음: 개발용 수치다. 판정에 쓰지 않는다.")
    lines.append("")

    lines += _table(["run_id", "crop", "seed", "프레임", "지문", "synth_commit"],
                    [[m["run_id"], m["crop"], str(m["seed"]), str(m["n_frames"]), m["fingerprint"][:12],
                      m["synth_commit"] or "-"] for m in (run["meta"] for run in results["runs"])])
    lines.append("")

    crop = results["crop"]
    for run in results["runs"]:
        multi = len(results["runs"]) > 1
        if multi:
            lines.append(f"## {run['meta']['run_id']}")
        for rung_name, rung in run["rungs"].items():
            lines.append(f"### {rung_name}" if not multi else f"#### {rung_name}")
            lines.append("")
            lines.append("**Y1**")
            lines += _table(_Y1_HEADERS, _y1_rows(rung["y1"]["passes"]))
            lines.append(_y1_summary_line(rung["y1"]["summary"]))
            lines.append("")
            y6 = rung["y6"]
            if y6 is not None:
                lines.append("**Y6**")
                lines += _table(_Y6_TREE_HEADERS, _y6_tree_rows(y6["trees"]))
                lines.append(_row_sum_line(y6["row"]))
                lines += _ledger_table(y6["ledger"], y6["ledger_any"])
                lines.append(_metrics_line(y6["metrics"]))
                lines.append("")

        if len(run.get("chain") or []) >= 2:
            lines.append("**기여**")
            lines += _contrib_table(run["chain"], run["rungs"])
            lines.append("")

        if run.get("interaction_2x2") is not None:
            lines.append("**2 × 2**")
            lines += _cell_table(run["interaction_2x2"]["cells"])
            lines.append("")

        if criteria is not None:
            entries = [(rung_name, criteria_rows(rung, criteria, rung_name, crop))
                       for rung_name, rung in run["rungs"].items()]
            entries = [(name, crows) for name, crows in entries if crows]
            if entries:
                lines.append("**기준과 비교(가정값, 참고)**")
                lines += _criteria_table(entries)
                lines.append("")

    return "\n".join(lines) + "\n"
