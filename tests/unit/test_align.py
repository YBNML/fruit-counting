"""Aligning the tracked passes without truth: the alignment knobs and draw steps, the segments along the
tree row, the rigid fits of a segment, the check that keeps the identity, the coarse shift of a group fit
between the sides and their fallbacks, the alignment of passes by the tracks two passes share and to the row
centre plane, and the stress options."""

import dataclasses
import inspect
import re
from math import radians

import numpy as np
import pytest
from pydantic import ValidationError
from scipy.spatial.transform import Rotation

import counting.geometry.noise as noise_module
from counting.data.formats.orchard_synth import OrchardLayout, PassKey
from counting.evaluation.alignment import ROLL_PITCH_SIGMA_START_DEG, align_settings, roll_pitch_sigma
from counting.geometry.noise import noise_rng
from counting.geometry.rig import rigid
from counting.multiview.align import (HUBER_M, MIN_PAIRS, PAIR_RADIUS_M, SEGMENT_M, AlignPass, AlignSettings,
                                      PassAlignment, Segments, align_passes, fit_known, fit_rigid, fit_segments,
                                      track_x)
from counting.odometry.imufile import GRAVITY
from tests.unit.synthfake import fake_cfg

pytestmark = pytest.mark.pose

LAYOUT = OrchardLayout(crop="apple", row_ys=(0.0, 3.75, 7.5), target_row=1, row_spacing=3.75,
                       tree_ids=tuple(range(12)), tree_x=tuple(1.25 * k for k in range(12)))


def _cloud(n, seed, x=(-1.0, 14.75), half=0.4, z=(0.8, 3.2)):
    """`n` fruit-like points around the row centre plane of `LAYOUT`."""
    rng = np.random.default_rng(seed)
    return np.c_[rng.uniform(*x, n), LAYOUT.row_y + rng.uniform(-half, half, n), rng.uniform(*z, n)]


def _move(G, X):
    return X @ G[:3, :3].T + G[:3, 3]


def test_alignment_settings_and_draw_steps(tmp_path):              # the configs name a run folder, never open it
    cfg = fake_cfg(tmp_path, "apple", "pose.rig_ref=30", "align.mode=[tracks,row_plane]", "align.segment_m=4",
                   "align.pair_radius_m=0.3", "align.roll_pitch_sigma_deg=1.0", "align.min_pairs=10",
                   "align.oracle_sigma_m=0.03")
    a = cfg.align
    assert (a.segment_m, a.pair_radius_m, a.roll_pitch_sigma_deg, a.min_pairs, a.oracle_sigma_m) == (4.0, 0.3, 1.0, 10, 0.03)
    bare = fake_cfg(tmp_path, "apple", "pose.rig_ref=30", "align.mode=[tracks]").align
    assert set(bare.model_dump()) == {"mode"}            # unset knobs stay out of the dump and the config hash
    # values a base config carries under mode none, the default (a --set value none reads as null)
    carried = fake_cfg(tmp_path, "apple", "pose.rig_ref=30", "align.segment_m=4", "align.pair_radius_m=0.6")
    assert (carried.align.mode, carried.align.segment_m, carried.align.pair_radius_m) == ("none", 4.0, 0.6)
    truth = fake_cfg(tmp_path, "apple", "pose.rig_ref=30", "align.mode=fit_truth_pass", "align.min_pairs=5")
    assert truth.align.min_pairs == 5                    # a mode that does not use a knob takes it all the same
    for sets, message in ((("align.segment_m=0",), "greater than 0"),
                          (("align.pair_radius_m=-0.1",), "greater than 0"),
                          (("align.roll_pitch_sigma_deg=0",), "greater than 0"),
                          (("align.min_pairs=2",), "greater than or equal to 3"),
                          (("align.oracle_sigma_m=0",), "greater than 0"),
                          (("align.mode=[row_plane]", "align.stress_pair_keep=0.5"),
                           "align.stress_pair_keep needs tracks in align.mode")):
        with pytest.raises(ValidationError, match=re.escape(message)):
            fake_cfg(tmp_path, "apple", "pose.rig_ref=30", "align.mode=[tracks,row_plane]", *sets)
    steps = (noise_module.STEP_ALIGN, noise_module.STEP_ALIGN_THIN, noise_module.STEP_STRUCT)
    others = {getattr(noise_module, n) for n in dir(noise_module)
              if n.startswith("STEP_") and n not in ("STEP_ALIGN", "STEP_ALIGN_THIN", "STEP_STRUCT")}
    assert steps == (15, 16, 17) and max(others) == 14 and not others & set(steps)


def test_segments_lie_along_the_tree_row():
    seg = Segments.along(LAYOUT.tree_x, SEGMENT_M)
    np.testing.assert_allclose(seg.centres, 1.5 * np.arange(10))
    assert seg.index(np.array([-5.0, 0.75, 0.76, 20.0, np.nan])).tolist() == [0, 0, 1, 9, 0]
    assert seg.window(1, np.array([-0.1, 0.0, 3.0, 3.1])).tolist() == [False, True, True, False]
    assert Segments.along([0.0, 1.0], 3.0).centres.tolist() == [0.0]
    whole = Segments.whole()
    assert whole.index(np.array([-1e3, 1e3])).tolist() == [0, 0] and whole.window(0, np.array([1e6])).all()
    assert (SEGMENT_M, PAIR_RADIUS_M, MIN_PAIRS, HUBER_M) == (3.0, 0.5, 20, 0.02)


def test_pass_alignment_moves_by_segment_and_composes():
    seg = Segments.along([0.0, 3.0], 3.0)                         # centres 0, 1.5 and 3
    shift = PassAlignment(seg, np.stack([rigid([0.0, 0.0, 0.0], [0.0, 0.1 * j, 0.0]) for j in range(3)]))
    X = np.array([[0.2, 0.0, 1.0], [1.4, 0.0, 1.0], [np.nan] * 3, [9.0, 0.0, 1.0]])
    x = np.array([0.2, 1.4, 1.4, 9.0])
    moved = shift.move(X, x)
    np.testing.assert_allclose(moved[[0, 1, 3]], [[0.2, 0.0, 1.0], [1.4, 0.1, 1.0], [9.0, 0.2, 1.0]], atol=1e-12)
    assert np.isnan(moved[2]).all()
    assert np.array_equal(PassAlignment.identity(seg).move(X, x), X, equal_nan=True)
    turn = PassAlignment(seg, np.stack([rigid([0.0, 0.0, 0.01], [0.0, 0.0, 0.0])] * 3))
    both = turn.after(shift)                                      # shift first, then turn
    np.testing.assert_allclose(both.move(X[:2], x[:2]), turn.move(shift.move(X[:2], x[:2]), x[:2]), atol=1e-12)
    np.testing.assert_allclose(both.at(np.array([1.4]))[0], turn.matrices[1] @ shift.matrices[1], atol=1e-12)
    with pytest.raises(ValueError, match="alignments on different segments"):
        turn.after(PassAlignment.identity(Segments.whole()))


def test_rigid_fit_finds_a_known_offset_without_known_pairs():
    X = _cloud(300, 3, x=(0.0, 3.0))
    error = rigid([radians(0.2), 0.0, radians(0.4)], [0.12, -0.08, 0.04])
    moving = _move(error, X[:240]) + np.random.default_rng(4).normal(0.0, 0.003, (240, 3))
    A, n, residual = fit_rigid(moving, X, radius_m=0.5, sigma_rp_deg=None, min_pairs=20, rng=noise_rng(1))
    assert n == 240 and residual < 0.008
    assert np.linalg.norm(_move(A, _move(error, X[:240])) - X[:240], axis=1).max() < 0.003
    assert np.degrees(Rotation.from_matrix((A @ error)[:3, :3]).magnitude()) < 0.05
    assert fit_rigid(moving, X, radius_m=0.5, sigma_rp_deg=None, min_pairs=241, rng=noise_rng(1))[:2] == (None, 240)
    assert fit_rigid(moving, X + [0.0, 0.0, 5.0], radius_m=0.5, sigma_rp_deg=None, min_pairs=20,
                     rng=noise_rng(1)) == (None, 0, None)


def test_rigid_fit_gathers_its_pairs_again_along_a_long_window():
    X = _cloud(300, 51, x=(0.0, 20.0))
    error = rigid([0.0, 0.0, radians(0.5)], [0.1, 0.05, 0.02])     # the yaw moves the far end 17 cm more
    moving = _move(error, X) + np.random.default_rng(52).normal(0.0, 0.02, X.shape)
    A, n, _ = fit_rigid(moving, X, radius_m=0.5, sigma_rp_deg=None, min_pairs=20, rng=noise_rng(1))
    assert n >= 225                                               # one shift alone keeps the middle pairs only
    assert np.percentile(np.linalg.norm(_move(A, _move(error, X)) - X, axis=1), 90) < 0.01


def test_roll_pitch_prior_holds_back_the_tilt():
    X = _cloud(100, 5, x=(0.0, 3.0))
    c = X.mean(axis=0)
    tilt = Rotation.from_euler("ZYX", [radians(0.3), radians(0.5), radians(1.0)]).as_matrix()
    moving = (X - c) @ tilt.T + c
    free, residual = fit_known(moving, X, sigma_rp_deg=None)
    assert residual < 1e-9
    np.testing.assert_allclose(free[:3, :3] @ tilt, np.eye(3), atol=1e-9)
    held, _ = fit_known(moving, X, sigma_rp_deg=0.01)
    yaw, pitch, roll = np.degrees(Rotation.from_matrix(held[:3, :3]).as_euler("ZYX"))
    assert abs(roll) < 0.05 and abs(pitch) < 0.05                 # the free fit turns back by about 1 and 0.5 degrees


def test_segment_fits_borrow_then_go_whole_then_identity():
    seg = Segments.along([0.0, 6.0], 3.0)                         # centres 0, 1.5, 3, 4.5 and 6
    X = np.r_[_cloud(30, 7, x=(-1.0, -0.2)), _cloud(5, 8, x=(6.2, 7.0))]
    moving = _move(rigid([0.0, 0.0, 0.0], [0.05, 0.02, 0.0]), X)

    def fits(min_pairs):
        return fit_segments(moving, moving[:, 0], X, seg, radius_m=0.5, sigma_rp_deg=None, min_pairs=min_pairs,
                            key=(0, 0, 15, 5))

    got = fits(20)
    assert [(f.n, f.use) for f in got] == [(30, "own"), (0, "nearest"), (0, "nearest"), (0, "nearest"), (5, "nearest")]
    assert all(np.array_equal(f.matrix, got[0].matrix) for f in got) and got[0].residual_m is not None
    assert [(f.n, f.use) for f in fits(32)] == [(30, "group"), (0, "group"), (0, "group"), (0, "group"), (5, "group")]
    lost = fits(40)
    assert [f.use for f in lost] == ["identity"] * 5
    assert all(f.residual_m is None and np.array_equal(f.matrix, np.eye(4)) for f in lost)
    assert all(a.matrix.tobytes() == b.matrix.tobytes() for a, b in zip(got, fits(20)))


def test_a_fit_must_beat_the_identity_on_held_out_pairs():
    for seed in range(5):                      # the same 30 points on both sides, noise only: the identity is right
        X = _cloud(150, 60 + seed, x=(0.0, 3.0))
        rng = np.random.default_rng(70 + seed)
        moving = np.r_[X[:30] + rng.normal(0.0, 0.015, (30, 3)), _cloud(120, 80 + seed, x=(0.0, 3.0))]
        reference = np.r_[X[:30] + rng.normal(0.0, 0.015, (30, 3)), X[30:]]
        A, n, residual = fit_rigid(moving, reference, radius_m=0.5, sigma_rp_deg=None, min_pairs=20,
                                   rng=noise_rng(1))
        assert A is None and residual is None and n >= 20       # enough pairs, refused
    error = rigid([0.0, 0.0, 0.0], [0.06, -0.03, 0.02])
    for seed in (0, 1):                                          # a real offset of 7 cm stands
        X = _cloud(150, 60 + seed, x=(0.0, 3.0))
        rng = np.random.default_rng(70 + seed)
        moving = np.r_[_move(error, X[:30]) + rng.normal(0.0, 0.015, (30, 3)), _cloud(120, 80 + seed, x=(0.0, 3.0))]
        A, n, _ = fit_rigid(moving, X, radius_m=0.5, sigma_rp_deg=None, min_pairs=20, rng=noise_rng(1))
        assert A is not None and n >= 20
        assert np.linalg.norm((_move(A, _move(error, X[:30])) - X[:30]).mean(axis=0)) < 0.01


def test_a_noise_fit_is_not_borrowed_along_the_pass():
    seg = Segments.along([0.0, 6.0], 3.0)                         # centres 0, 1.5, 3, 4.5 and 6
    for seed in range(3):
        X = np.r_[_cloud(25, 7 + seed, x=(-1.0, -0.2)), _cloud(5, 8 + seed, x=(6.2, 7.0))]
        rng = np.random.default_rng(20 + seed)
        moving, reference = X + rng.normal(0.0, 0.01, X.shape), X + rng.normal(0.0, 0.01, X.shape)
        fits = fit_segments(moving, moving[:, 0], reference, seg, radius_m=0.5, sigma_rp_deg=None, min_pairs=20,
                            key=(0, 0, 15, 5))
        assert [f.use for f in fits] == ["identity"] * 5 and fits[0].n >= 20
        assert all(f.residual_m is None and np.array_equal(f.matrix, np.eye(4)) for f in fits)


def test_the_group_fit_between_the_sides_retries_after_a_coarse_shift():
    seg = Segments.along([0.0, 20.0], 3.0)                        # 14 segments
    X = _cloud(300, 51, x=(0.0, 20.0))
    error = rigid([0.0, 0.0, radians(0.1)], [0.8, 0.3, -0.1])     # beyond the pair radius everywhere
    moving = _move(error, X[:240]) + np.random.default_rng(52).normal(0.0, 0.01, (240, 3))

    def fits(coarse, M=moving, x=moving[:, 0], R=X):
        return fit_segments(M, x, R, seg, radius_m=0.5, sigma_rp_deg=None, min_pairs=20,
                            key=(0, 0, 15, 5, 1, 1, 0), coarse=coarse)

    shifted = fits(True)
    assert [f.use for f in shifted] == ["group"] * len(seg.centres)
    assert np.percentile(np.linalg.norm(_move(shifted[0].matrix, _move(error, X[:240])) - X[:240], axis=1), 90) < 0.005
    assert [f.use for f in fits(False)] == ["identity"] * len(seg.centres)     # no coarse shift on one side
    for seed in (0, 1):                                           # two groups with no fruit in common stay apart
        low = _cloud(300, 90 + seed, x=(0.0, 20.0), z=(0.8, 2.0))
        high = _cloud(300, 95 + seed, x=(0.0, 20.0), z=(1.8, 3.2))
        assert {f.use for f in fits(True, high, high[:, 0], low)} == {"identity"}


def _split(X, side, error, seed, aisle):
    """The two passes of one side: pitch 10 sees the points of `X` below 2.0 m, pitch 50 those from 1.6 m up (both
    see 1.6 to 2.0 m); every point moved by `error`, with 3 mm noise. Returns (pass, its true points) pairs."""
    seen = _move(error, X) + np.random.default_rng(seed).normal(0.0, 0.003, X.shape)
    out = []
    for pitch, sel in ((10.0, X[:, 2] < 2.0), (50.0, X[:, 2] >= 1.6)):
        P = seen[sel]
        out.append((AlignPass(PassKey(pitch, side, aisle), P, np.full(len(P), 3), P[:, 0].copy()), X[sel]))
    return out


def _error_p90(alignment, p, truth):
    return float(np.percentile(np.linalg.norm(alignment.move(p.X, p.x) - truth, axis=1), 90))


def test_tracks_bring_side_b_onto_side_a_under_the_rig():
    X = _cloud(900, 11)
    error = rigid([radians(0.2), 0.0, radians(0.3)], [0.1, 0.05, 0.02])
    items = _split(X, "A", np.eye(4), 12, 2) + _split(X, "B", error, 13, 1)
    result = align_passes([p for p, _ in items], LAYOUT, ["tracks"], AlignSettings(), rig=True, ref_pitch=50.0,
                          key=(0, 0, 5))
    for p, truth in items:
        assert _error_p90(result.transforms[p.key], p, truth) < 0.012, p.key
    for p, _ in items[:2]:
        assert np.array_equal(result.transforms[p.key].matrices, np.repeat(np.eye(4)[None], 10, axis=0))
    b10, b50 = (result.transforms[p.key] for p, _ in items[2:])
    assert np.array_equal(b10.matrices, b50.matrices)             # one body: one transform for both cameras
    (stage,) = result.stages
    assert (stage.stage, stage.group, [f.use for f in stage.fits]) == ("both_sides", "B", ["own"] * 10)
    assert min(f.n for f in stage.fits) >= 100


def test_without_the_rig_each_pass_is_its_own_group():
    X = _cloud(900, 21)
    a10 = _split(X, "A", rigid([0.0, radians(0.3), 0.0], [0.02, 0.0, 0.03]), 23, 2)[0]
    a50 = _split(X, "A", np.eye(4), 22, 2)[1]
    b10 = _split(X, "B", rigid([0.0, 0.0, radians(0.2)], [0.08, 0.03, 0.0]), 24, 1)[0]
    b50 = _split(X, "B", rigid([radians(0.2), 0.0, 0.0], [0.12, -0.02, 0.02]), 25, 1)[1]
    items = [a10, a50, b10, b50]
    result = align_passes([p for p, _ in items], LAYOUT, ["tracks"], AlignSettings(), rig=False, ref_pitch=50.0,
                          key=(0, 0, 5))
    for p, truth in items:
        assert _error_p90(result.transforms[p.key], p, truth) < 0.012, p.key
    assert [(s.stage, s.group) for s in result.stages] == [("same_side", "10A"), ("same_side", "10B"),
                                                          ("both_sides", "10B"), ("both_sides", "50B")]
    with pytest.raises(ValueError, match="unknown alignment method: trees"):
        align_passes([p for p, _ in items], LAYOUT, ["trees"], AlignSettings(), rig=False, ref_pitch=50.0, key=(0, 0, 5))


def test_row_plane_moves_both_sides_to_the_row_centre():
    rng = np.random.default_rng(31)
    xa, xb = rng.uniform(-1.0, 14.75, 600), rng.uniform(-1.0, 14.75, 200)   # three times as many tracks on side A
    ya, yb = LAYOUT.row_y + rng.uniform(0.05, 0.4, 600), LAYOUT.row_y - rng.uniform(0.05, 0.4, 200)
    za, zb = rng.uniform(0.8, 3.2, 600), rng.uniform(0.8, 3.2, 200)

    def off(x):                                                   # a common side error that grows along the row
        return 0.06 + 0.004 * x

    side_a = AlignPass(PassKey(50.0, "A", 2), np.c_[xa, ya + off(xa), za], np.full(600, 3), xa)
    side_b = AlignPass(PassKey(50.0, "B", 1), np.c_[xb, yb + off(xb), zb], np.full(200, 3), xb)
    centres = Segments.along(LAYOUT.tree_x, SEGMENT_M).centres
    for bias in (0.0, 0.05):
        result = align_passes([side_a, side_b], LAYOUT, ["row_plane"], AlignSettings(row_bias_m=bias), rig=True,
                              ref_pitch=50.0, key=(0, 0, 5))
        (stage,) = result.stages
        shifts = np.array([f.matrix[1, 3] for f in stage.fits])
        assert np.abs(shifts + off(centres) + bias).max() < 0.015
        assert (stage.stage, stage.group, [f.use for f in stage.fits]) == ("row_plane", "AB", ["own"] * 10)
        assert all(np.array_equal(np.delete(f.matrix, 1, axis=0)[:, 3], [0.0, 0.0, 1.0])
                   and np.array_equal(f.matrix[:3, :3], np.eye(3)) for f in stage.fits)
        assert np.array_equal(result.transforms[side_a.key].matrices, result.transforms[side_b.key].matrices)
    alone = align_passes([side_a], LAYOUT, ["row_plane"], AlignSettings(), rig=True, ref_pitch=50.0, key=(0, 0, 5))
    assert [f.use for f in alone.stages[0].fits] == ["identity"] * 10
    assert np.array_equal(alone.transforms[side_a.key].matrices, np.repeat(np.eye(4)[None], 10, axis=0))


def test_pair_keep_thins_the_moving_tracks():
    X = _cloud(900, 41)
    items = _split(X, "A", np.eye(4), 42, 2) + _split(X, "B", rigid([0.0, 0.0, radians(0.3)], [0.1, 0.05, 0.02]), 43, 1)

    def run(keep):
        result = align_passes([p for p, _ in items], LAYOUT, ["tracks"], AlignSettings(pair_keep=keep), rig=True,
                              ref_pitch=50.0, key=(0, 0, 5))
        return result, sum(f.n for f in result.stages[0].fits)

    _, n_full = run(None)
    half, n_half = run(0.5)
    again, _ = run(0.5)
    assert 0.35 < n_half / n_full < 0.65
    assert all(a.matrix.tobytes() == b.matrix.tobytes() for a, b in zip(half.stages[0].fits, again.stages[0].fits))


def test_alignment_inputs_carry_no_pose():
    assert {f.name for f in dataclasses.fields(AlignPass)} == {"key", "X", "n_obs", "x"}
    assert list(inspect.signature(align_passes).parameters) == ["passes", "layout", "methods", "settings", "rig",
                                                                 "ref_pitch", "key"]
    frames = [frozenset({3, 1}), frozenset({2}), frozenset()]
    np.testing.assert_allclose(track_x(frames, {1: 0.0, 2: 5.0, 3: 1.0}), [0.5, 5.0, np.nan])


def test_roll_pitch_prior_is_the_fixed_belief_under_fusion(tmp_path):
    assert roll_pitch_sigma(4.0, 0.3) == pytest.approx(np.degrees(2 * np.hypot(4e-3, radians(0.3))), rel=1e-12)
    assert ROLL_PITCH_SIGMA_START_DEG == roll_pitch_sigma(4.0, 0.3) == pytest.approx(0.7550, abs=1e-4)
    assert roll_pitch_sigma(0.0, 0.0) == pytest.approx(np.degrees(2 * 1e-4 / GRAVITY), rel=1e-12)   # the bias floor
    fused = ("pose.rig=true", "pose.rig_ref=30", "pose.fuse=imu", "align.mode=[tracks,row_plane]")
    for stress in ((), ("noise.imu_acc_b0_mg=40", "noise.imu_extr_rot_deg=1.0")):   # the IMU noise a config generates
        held = align_settings(fake_cfg(tmp_path, "apple", *fused, *stress))
        assert held.roll_pitch_sigma_deg == ROLL_PITCH_SIGMA_START_DEG, stress
    chosen = align_settings(fake_cfg(tmp_path, "apple", *fused, "align.roll_pitch_sigma_deg=1.0"))
    assert chosen.roll_pitch_sigma_deg == 1.0
    unfused = align_settings(fake_cfg(tmp_path, "apple", "pose.rig=true", "pose.rig_ref=30", "align.mode=[tracks,row_plane]",
                                      "align.roll_pitch_sigma_deg=1.0", "align.stress_row_bias_m=0.05"))
    assert (unfused.roll_pitch_sigma_deg, unfused.segment_m, unfused.pair_radius_m, unfused.min_pairs, unfused.min_obs,
            unfused.row_bias_m, unfused.pair_keep) == (None, 3.0, 0.5, 20, 1, 0.05, None)   # no gravity, no prior
