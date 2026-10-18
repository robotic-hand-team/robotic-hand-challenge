"""
Two-stage session matcher for real-world grasp data.

Stage 1  — XGBoost classifies sensor type (sensor_A / sensor_B) from the
           mean tactile features over the hold phase.  LOO accuracy: 81.8% (27/33).
Stage 2  — Multivariate DTW finds the most similar session *within* the
           predicted sensor class.

Usage:
    python final_model/match_session.py  <query_session_path>
                                         [--lib  <dir>]
                                         [--top  <k>]
                                         [--rebuild-index]

    --lib            reference library  (default: ~/Documents/20261017_17October_8pm)
    --top            how many best matches to show  (default: 5)
    --rebuild-index  force re-labelling all library sessions
                     (runs automatically if the index file is missing)

Example:
    python final_model/match_session.py  ~/Documents/20261017_17October_8pm/20261017_202049
"""

import sys
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

# ── Project path setup ────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from matcher.match_session import extract_series, dtw_distance, detect_hold_window

# ── Paths ─────────────────────────────────────────────────────────────────────
CLASSIFIER  = Path(__file__).parent / "sensor_classifier.pkl"
DEFAULT_LIB = Path.home() / "Documents" / "20261017_17October_8pm"
INDEX_FILE  = Path(__file__).parent / "library_index.pkl"

ANGLE_COLS = [
    "angle_act_little", "angle_act_ring", "angle_act_middle",
    "angle_act_index", "angle_act_thumb_bend", "angle_act_thumb_rot",
]
POS_ACT_COLS = [
    "pos_act_little", "pos_act_ring", "pos_act_middle",
    "pos_act_index", "pos_act_thumb_bend",
]
MIN_HOLD_FRAMES = 15


# ── Feature extraction for classifier ────────────────────────────────────────

def _hold_mean_features(session_path: Path, feature_cols: list[str]) -> np.ndarray | None:
    tac_path = session_path / "tactile_summary.csv"
    act_path = session_path / "actuators.csv"
    if not tac_path.exists() or not act_path.exists():
        return None

    tac     = pd.read_csv(tac_path)
    act_all = pd.read_csv(act_path)
    act     = act_all[["frame"] + [c for c in ANGLE_COLS + POS_ACT_COLS if c in act_all.columns]]
    df      = tac.merge(act, on="frame").fillna(0)

    if len(df) < MIN_HOLD_FRAMES:
        return None

    h_start, h_end = detect_hold_window(df)
    hold = df.iloc[h_start:h_end]
    if len(hold) < MIN_HOLD_FRAMES:
        hold = df

    return np.array(
        [hold[c].mean() if c in hold.columns else 0.0 for c in feature_cols],
        dtype=np.float32,
    )


def predict_sensor(session_path: Path, xgb, scaler, feature_cols: list[str], le) -> str | None:
    """Return 'sensor_A' or 'sensor_B' for the session, or None if data missing."""
    feats = _hold_mean_features(session_path, feature_cols)
    if feats is None:
        return None
    X = scaler.transform(feats.reshape(1, -1))
    idx = xgb.predict(X)[0]
    return str(le.inverse_transform([idx])[0])


# ── Library index ─────────────────────────────────────────────────────────────

def build_index(lib_dir: Path, xgb, scaler, feature_cols: list[str], le) -> dict:
    sessions = sorted(p for p in lib_dir.iterdir() if p.is_dir())
    index: dict[str, list] = {"sensor_A": [], "sensor_B": []}
    skipped = 0

    for s in sessions:
        label     = predict_sensor(s, xgb, scaler, feature_cols, le)
        ts_result = extract_series(s)
        if label is None or ts_result is None or label not in index:
            skipped += 1
            continue
        index[label].append((s, ts_result[0]))

    print(
        f"  Library indexed: "
        f"{len(index['sensor_A'])}× sensor_A, "
        f"{len(index['sensor_B'])}× sensor_B  "
        f"({skipped} skipped)"
    )
    return index


def load_or_build_index(lib_dir: Path, xgb, scaler, feature_cols, le, rebuild: bool) -> dict:
    if not rebuild and INDEX_FILE.exists():
        with open(INDEX_FILE, "rb") as f:
            cached = pickle.load(f)
        if cached.get("lib_dir") == str(lib_dir):
            print(f"  Using cached index: {INDEX_FILE.name}")
            return cached["index"]

    print("  Building library index (this takes ~10 s)…")
    index = build_index(lib_dir, xgb, scaler, feature_cols, le)
    with open(INDEX_FILE, "wb") as f:
        pickle.dump({"lib_dir": str(lib_dir), "index": index}, f)
    return index


# ── Two-stage matching ────────────────────────────────────────────────────────

def find_matches_dtw(
    query_ts: np.ndarray,
    candidates: list[tuple[Path, np.ndarray]],
    top_k: int,
) -> list[tuple[float, Path]]:
    paths  = [c[0] for c in candidates]
    series = [c[1] for c in candidates]

    with ThreadPoolExecutor(max_workers=8) as ex:
        dists = list(ex.map(lambda b: dtw_distance(query_ts, b), series))

    order = np.argsort(dists)[:top_k]
    return [(dists[i], paths[i]) for i in order]


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Two-stage session matcher (PCA+SVM + DTW)")
    ap.add_argument("query",           type=Path, help="Query session folder")
    ap.add_argument("--lib",           type=Path, default=DEFAULT_LIB)
    ap.add_argument("--top",           type=int,  default=5)
    ap.add_argument("--rebuild-index", action="store_true")
    args = ap.parse_args()

    query_path = args.query.expanduser().resolve()
    lib_dir    = args.lib.expanduser().resolve()

    if not query_path.is_dir():
        sys.exit(f"ERROR: query path not found: {query_path}")
    if not lib_dir.is_dir():
        sys.exit(f"ERROR: library dir not found: {lib_dir}")

    # Load bundled classifier (XGBoost retrained on 33 labeled real-world sessions)
    with open(CLASSIFIER, "rb") as f:
        clf = pickle.load(f)
    xgb, scaler, feature_cols, le = clf["xgb"], clf["scaler"], clf["feature_cols"], clf["label_encoder"]

    # ── Stage 1: classify query sensor ───────────────────────────────────────
    print(f"\nQuery   : {query_path.name}")
    sensor_pred = predict_sensor(query_path, xgb, scaler, feature_cols, le)
    if sensor_pred is None:
        sys.exit("ERROR: could not extract features from query session")
    print(f"  XGBoost prediction → {sensor_pred}")

    # ── Extract DTW series for query ──────────────────────────────────────────
    ts_result = extract_series(query_path)
    if ts_result is None:
        sys.exit("ERROR: could not extract time series from query session")
    query_ts, h_start, h_end = ts_result
    print(f"  Hold window: frames {h_start}–{h_end}  →  "
          f"{query_ts.shape[0]} frames × {query_ts.shape[1]} channels")

    # ── Load / build library index ────────────────────────────────────────────
    print(f"\nLibrary : {lib_dir}")
    index = load_or_build_index(lib_dir, xgb, scaler, feature_cols, le, args.rebuild_index)

    # ── Stage 2: DTW within predicted class ───────────────────────────────────
    candidates = [c for c in index[sensor_pred] if c[0].resolve() != query_path]
    print(f"\n  DTW search within {sensor_pred}: {len(candidates)} candidates…")

    if not candidates:
        sys.exit("ERROR: no library sessions in predicted class (try --rebuild-index)")

    matches = find_matches_dtw(query_ts, candidates, args.top)

    print(f"\n{'Rank':<5}  {'DTW dist':>10}  Session")
    print("-" * 58)
    for rank, (dist, path) in enumerate(matches, 1):
        marker = "  ← best match" if rank == 1 else ""
        print(f"  {rank:<3}  {dist:>10.2f}  {path.name}{marker}")

    best_path = matches[0][1]
    print(f"\nBest match: {best_path}")
    return best_path


if __name__ == "__main__":
    main()
