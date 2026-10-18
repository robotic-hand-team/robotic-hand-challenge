"""
Session matcher with time-series DTW.

Given an unknown session folder, find the most similar session in a
reference library using multivariate Dynamic Time Warping (DTW) on the
hold-phase tactile + joint-angle time series.

Usage:
    python matcher/match_session.py  <query_session_path>  [--lib <dir>]  [--top <k>]

    --lib   reference library (default: ~/Documents/20261017_17October_8pm)
    --top   how many best matches to return (default: 5)

Example:
    python matcher/match_session.py ~/Documents/20261017_17October_8pm/20261017_202049

How it works:
    1. Detect the hold phase automatically from actuator stability
       (longest window where all finger positions change < 20 units / frame).
       Falls back to middle 50 % if detection fails.
    2. Build a 12-d time series from the hold window:
         channels 0–5  : per-finger total activation + palm  (weight × 3)
         channels 6–11 : joint angles                        (weight × 1)
       Tactile channels are up-weighted 3 × so the DTW path prioritises
       contact pattern over pose similarity.
    3. Downsample to ≤ 60 frames, z-score normalise per channel.
    4. Multivariate DTW distance via dtaidistance.dtw_ndim.
    5. Return top-k sessions, parallelised over CPU cores.
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from dtaidistance import dtw_ndim


# ── Constants ─────────────────────────────────────────────────────────────────

ANGLE_COLS = [
    "angle_act_little", "angle_act_ring", "angle_act_middle",
    "angle_act_index", "angle_act_thumb_bend", "angle_act_thumb_rot",
]
POS_ACT_COLS = [
    "pos_act_little", "pos_act_ring", "pos_act_middle",
    "pos_act_index", "pos_act_thumb_bend",
]
FINGER_PADS = {
    "little": ["little_tip_sum", "little_nail_sum", "little_pad_sum"],
    "ring":   ["ring_tip_sum",   "ring_nail_sum",   "ring_pad_sum"],
    "middle": ["middle_tip_sum", "middle_nail_sum", "middle_pad_sum"],
    "index":  ["index_tip_sum",  "index_nail_sum",  "index_pad_sum"],
    "thumb":  ["thumb_tip_sum",  "thumb_nail_sum",  "thumb_pad_sum"],
}
PALM_COL = "palm_sum"

TACTILE_WEIGHT = 3.0   # up-weight tactile vs joint angles
DOWNSAMPLE_TO  = 60    # max frames after resampling
STABILITY_STD  = 20.0  # pos_act rolling-std threshold for hold detection
STABILITY_WIN  = 10    # rolling window size (frames)
MIN_HOLD_FRAMES = 15


# ── Hold-phase detection ──────────────────────────────────────────────────────

def detect_hold_window(act_df: pd.DataFrame) -> tuple[int, int]:
    """
    Return (start, end) indices of the longest stable hold window.
    Stability = rolling std of all pos_act columns < STABILITY_STD.
    Falls back to middle 50 % of the recording if detection fails.
    """
    available = [c for c in POS_ACT_COLS if c in act_df.columns]
    if not available:
        n = len(act_df)
        return n // 4, n - n // 4

    pos  = act_df[available]
    rstd = pos.rolling(STABILITY_WIN, center=True).std().fillna(9999).max(axis=1)
    stable = (rstd < STABILITY_STD).values

    # Find the longest contiguous stable run
    transitions = np.diff(stable.astype(int))
    starts = np.where(transitions == 1)[0] + 1
    ends   = np.where(transitions == -1)[0] + 1
    if stable[0]:
        starts = np.concatenate([[0], starts])
    if stable[-1]:
        ends = np.concatenate([ends, [len(stable)]])

    if len(starts) == 0 or (len(starts) > 0 and (ends - starts).max() < MIN_HOLD_FRAMES):
        n = len(act_df)
        return n // 4, n - n // 4

    best = np.argmax(ends - starts)
    return int(starts[best]), int(ends[best])


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_series(session_path: Path) -> tuple[np.ndarray, int, int] | None:
    """
    Returns (float32 array of shape (T, 12), hold_start, hold_end) or None.

    Channel layout (after weighting):
      0–5  : per-finger activation (little/ring/middle/index/thumb) + palm  × TACTILE_WEIGHT
      6–11 : joint angles                                                    × 1
    """
    tac_path = session_path / "tactile_summary.csv"
    act_path = session_path / "actuators.csv"
    if not tac_path.exists() or not act_path.exists():
        return None

    tac     = pd.read_csv(tac_path)
    act_all = pd.read_csv(act_path)
    act     = act_all[["frame"] + [c for c in ANGLE_COLS + POS_ACT_COLS if c in act_all.columns]]
    df      = tac.merge(act, on="frame").fillna(0)

    if len(df) < MIN_HOLD_FRAMES * 2:
        return None

    # Detect hold window from actuator stability
    h_start, h_end = detect_hold_window(df)
    hold = df.iloc[h_start:h_end].reset_index(drop=True)

    if len(hold) < MIN_HOLD_FRAMES:
        hold = df  # fallback: use everything

    # Per-finger total activation (tactile channels)
    finger_series = []
    for cols in FINGER_PADS.values():
        available = [c for c in cols if c in hold.columns]
        finger_series.append(
            hold[available].sum(axis=1).values if available else np.zeros(len(hold))
        )
    palm = hold[PALM_COL].values if PALM_COL in hold.columns else np.zeros(len(hold))

    # Joint angles
    angles = hold[[c for c in ANGLE_COLS if c in hold.columns]].values

    # Stack (T, 12) and apply weights
    tactile = np.column_stack(finger_series + [palm]) * TACTILE_WEIGHT
    ts      = np.column_stack([tactile, angles])

    # Downsample to DOWNSAMPLE_TO frames
    if len(ts) > DOWNSAMPLE_TO:
        idx = np.linspace(0, len(ts) - 1, DOWNSAMPLE_TO).astype(int)
        ts  = ts[idx]

    # Per-channel z-score
    mu  = ts.mean(axis=0)
    std = ts.std(axis=0)
    std[std < 1e-9] = 1.0
    ts  = (ts - mu) / std

    return ts.astype(np.float32), h_start, h_end


# ── Library ───────────────────────────────────────────────────────────────────

def build_library(lib_dir: Path) -> tuple[list[Path], list[np.ndarray]]:
    sessions = sorted(p for p in lib_dir.iterdir() if p.is_dir())
    paths, series = [], []
    for s in sessions:
        result = extract_series(s)
        if result is not None:
            paths.append(s)
            series.append(result[0])
    if not series:
        raise RuntimeError(f"No valid sessions in {lib_dir}")
    return paths, series


# ── DTW matching ──────────────────────────────────────────────────────────────

def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    return dtw_ndim.distance(a, b)


def find_matches(
    query_ts: np.ndarray,
    lib_paths: list[Path],
    lib_series: list[np.ndarray],
    top_k: int,
) -> list[tuple[float, Path]]:
    from multiprocessing import Pool, cpu_count
    from functools import partial

    with Pool(min(cpu_count(), 8)) as pool:
        dists = pool.map(partial(dtw_distance, query_ts), lib_series)

    order = np.argsort(dists)[:top_k]
    return [(dists[i], lib_paths[i]) for i in order]


# ── CLI ───────────────────────────────────────────────────────────────────────

DEFAULT_LIB = Path.home() / "Documents" / "20261017_17October_8pm"


def main():
    ap = argparse.ArgumentParser(description="Time-series DTW session matcher")
    ap.add_argument("query", type=Path, help="Query session folder")
    ap.add_argument("--lib",  type=Path, default=DEFAULT_LIB)
    ap.add_argument("--top",  type=int,  default=5)
    args = ap.parse_args()

    query_path = args.query.expanduser().resolve()
    lib_dir    = args.lib.expanduser().resolve()

    if not query_path.is_dir():
        sys.exit(f"ERROR: query path not found: {query_path}")
    if not lib_dir.is_dir():
        sys.exit(f"ERROR: library dir not found: {lib_dir}")

    print(f"Query   : {query_path.name}  (loading...)")
    result = extract_series(query_path)
    if result is None:
        sys.exit("ERROR: could not load query session")
    query_ts, h_start, h_end = result
    print(f"          hold window: frames {h_start}–{h_end}  →  {query_ts.shape[0]} frames × {query_ts.shape[1]} channels")

    print(f"Library : {lib_dir}\n")
    lib_paths, lib_series = build_library(lib_dir)

    filtered = [(p, s) for p, s in zip(lib_paths, lib_series)
                if p.resolve() != query_path]
    lib_paths_f  = [x[0] for x in filtered]
    lib_series_f = [x[1] for x in filtered]
    print(f"Library size: {len(lib_paths_f)} sessions  |  matching via DTW...\n")

    matches = find_matches(query_ts, lib_paths_f, lib_series_f, args.top)

    print(f"{'Rank':<5}  {'DTW dist':>10}  Session")
    print("-" * 55)
    for rank, (dist, path) in enumerate(matches, 1):
        marker = "  ← best match" if rank == 1 else ""
        print(f"  {rank:<3}  {dist:>10.2f}  {path.name}{marker}")

    print(f"\nBest match: {matches[0][1]}")
    return matches[0][1]


if __name__ == "__main__":
    main()
