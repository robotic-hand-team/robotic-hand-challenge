"""
XGBoost sensor classifier: Sensor A vs Sensor B
Leave-2-sessions-per-class cross-validation (all C(7,2) x C(6,2) = 315 combos)
"""

import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Session labels ────────────────────────────────────────────────────────────
DATASET_DIR = Path(__file__).parent / "dataset"

SESSIONS_A = [
    "20261016_204344",
    "20261016_204402",
    "20261016_204429",
    "20261016_204457",
    "20261016_204513",
    "20261016_204524",
    "20261016_204548",
]
SESSIONS_B = [
    "20261016_204558",
    "20261016_204609",
    "20261016_204619",
    "20261016_204627",
    "20261016_204637",
    "20261016_204649",
]

META_COLS = {"frame", "t_unix", "t_rel"}
ACTUATOR_FEATS = [
    "angle_act_little", "angle_act_ring", "angle_act_middle",
    "angle_act_index", "angle_act_thumb_bend", "angle_act_thumb_rot",
]


def load_session(session: str, label: int) -> pd.DataFrame:
    path = DATASET_DIR / session
    tac = pd.read_csv(path / "tactile_summary.csv")
    act = pd.read_csv(path / "actuators.csv")[["frame"] + ACTUATOR_FEATS]
    df = tac.merge(act, on="frame")
    df["label"] = label
    df["session"] = session
    # Drop meta, fill NaN centroids with 0
    feature_cols = [c for c in df.columns if c not in META_COLS | {"label", "session"}]
    df[feature_cols] = df[feature_cols].fillna(0)
    return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS | {"label", "session"}]


# ── Load all labeled sessions ─────────────────────────────────────────────────
print("Loading sessions...")
frames = []
for s in SESSIONS_A:
    frames.append(load_session(s, 0))
for s in SESSIONS_B:
    frames.append(load_session(s, 1))

all_data = pd.concat(frames, ignore_index=True)
feature_cols = get_feature_cols(all_data)
print(f"  {len(all_data)} total frames, {len(feature_cols)} features")
print(f"  Sensor A sessions: {len(SESSIONS_A)}, Sensor B sessions: {len(SESSIONS_B)}")

# ── Cross-validation ──────────────────────────────────────────────────────────
combos_a = list(itertools.combinations(SESSIONS_A, 2))   # C(7,2) = 21
combos_b = list(itertools.combinations(SESSIONS_B, 2))   # C(6,2) = 15
total_combos = len(combos_a) * len(combos_b)
print(f"\nRunning {len(combos_a)} x {len(combos_b)} = {total_combos} CV splits...")

results = []

for i, (test_a, test_b) in enumerate(itertools.product(combos_a, combos_b)):
    test_sessions  = set(test_a) | set(test_b)
    train_sessions = set(SESSIONS_A + SESSIONS_B) - test_sessions

    train = all_data[all_data["session"].isin(train_sessions)]
    test  = all_data[all_data["session"].isin(test_sessions)]

    X_train = train[feature_cols].values
    y_train = train["label"].values
    X_test  = test[feature_cols].values
    y_test  = test["label"].values

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        verbosity=0,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1  = f1_score(y_test, preds, average="binary")

    # Per-session accuracy
    per_session_acc = {}
    for s in test_sessions:
        mask = test["session"] == s
        per_session_acc[s] = accuracy_score(y_test[mask.values], preds[mask.values])

    results.append({
        "combo": i,
        "test_a": test_a,
        "test_b": test_b,
        "acc":    acc,
        "f1":     f1,
        **{f"acc_{s}": v for s, v in per_session_acc.items()},
    })

    if (i + 1) % 50 == 0:
        so_far = [r["acc"] for r in results]
        print(f"  [{i+1}/{total_combos}]  running mean acc = {np.mean(so_far):.3f}")

# ── Summary ───────────────────────────────────────────────────────────────────
accs = np.array([r["acc"] for r in results])
f1s  = np.array([r["f1"]  for r in results])

print("\n" + "="*55)
print("  XGBoost  —  Sensor A vs B  —  315-combo CV Results")
print("="*55)
print(f"  Accuracy:  mean={accs.mean():.3f}  std={accs.std():.3f}  "
      f"min={accs.min():.3f}  max={accs.max():.3f}")
print(f"  F1 score:  mean={f1s.mean():.3f}  std={f1s.std():.3f}")

# Distribution buckets
thresholds = [0.70, 0.80, 0.90, 0.95, 1.00]
print("\n  Accuracy distribution:")
prev = 0.0
for t in thresholds:
    count = ((accs >= prev) & (accs < t)).sum()
    print(f"    [{prev:.0%} – {t:.0%})  {count:4d} / {total_combos}  splits")
    prev = t

# Worst 5 combos
print("\n  Worst 5 splits:")
worst_idx = np.argsort(accs)[:5]
for idx in worst_idx:
    r = results[idx]
    print(f"    acc={r['acc']:.3f}  test_A={r['test_a']}  test_B={r['test_b']}")

# Best 5 combos
print("\n  Best 5 splits:")
best_idx = np.argsort(accs)[-5:][::-1]
for idx in best_idx:
    r = results[idx]
    print(f"    acc={r['acc']:.3f}  test_A={r['test_a']}  test_B={r['test_b']}")

# Per-session mean accuracy (how hard is each session to classify?)
print("\n  Per-session mean accuracy when in test set:")
for s in SESSIONS_A + SESSIONS_B:
    col = f"acc_{s}"
    vals = [r[col] for r in results if col in r]
    label = "A" if s in SESSIONS_A else "B"
    print(f"    [{label}] {s}  mean={np.mean(vals):.3f}  min={np.min(vals):.3f}")

# Save results
res_df = pd.DataFrame([{"combo": r["combo"], "acc": r["acc"], "f1": r["f1"],
                         "test_a_0": r["test_a"][0], "test_a_1": r["test_a"][1],
                         "test_b_0": r["test_b"][0], "test_b_1": r["test_b"][1]}
                        for r in results])
out_path = Path(__file__).parent / "cv_results.csv"
res_df.to_csv(out_path, index=False)
print(f"\n  Full results saved → {out_path}")
