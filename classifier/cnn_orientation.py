"""
2D-CNN orientation classifier on raw tactile data.
Input: 10 patch channels (nail + pad sensors), each 12×8 per frame.
Runs on Apple MPS (M2) for fast training.

Uses ALL phases (approach + grasp + hold) — not just hold.
Labels based on PITCH (tilt from vertical), not roll (cylinder axis):
  Class 0: vertical   |pitch| <  30°
  Class 1: tilted     30° ≤ |pitch| < 60°
  Class 2: horizontal |pitch| ≥  60°

Split: 80/20 session level
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.transform import Rotation
from sklearn.metrics import accuracy_score, classification_report

HERE      = Path(__file__).parent
SIM_DIR   = HERE.parent / "models" / "dataset_sim"
MODEL_DIR = HERE / "model"
MODEL_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# 12×8 patches → used as-is
PATCHES_12x8 = ["little_nail", "ring_nail", "middle_nail",
                "index_nail", "thumb_nail", "thumb_pad"]
# 10×8 patches → zero-pad 2 rows to 12×8
PATCHES_10x8 = ["little_pad", "ring_pad", "middle_pad", "index_pad"]

ALL_PATCHES = PATCHES_12x8 + PATCHES_10x8   # 10 channels
PATCH_H, PATCH_W = 12, 8
N_CHANNELS = len(ALL_PATCHES)


# ── Helpers ───────────────────────────────────────────────────────────────────
def patch_cols(raw_cols: list[str], patch: str, H: int, W: int) -> list[str]:
    cols = [(c, int(c.split("_r")[1].split("_c")[0]), int(c.split("_c")[1]))
            for c in raw_cols if c.startswith(patch + "_r")]
    cols.sort(key=lambda x: (x[1], x[2]))
    assert len(cols) == H * W, f"{patch}: expected {H*W}, got {len(cols)}"
    return [c[0] for c in cols]


CLASS_NAMES = ["vertical (|p|<30°)", "tilted (30-60°)", "horizontal (|p|≥60°)"]
N_CLASSES   = 3

def pitch_to_class(pitch_deg: float) -> int:
    p = abs(pitch_deg)
    if   p <  30: return 0   # vertical
    elif p <  60: return 1   # tilted
    else:         return 2   # horizontal


def session_to_labels(session_path: Path) -> pd.Series | None:
    """Returns per-frame label Series based on pitch, all phases."""
    pose = pd.read_csv(session_path / "object_pose.csv")
    if len(pose) < 5:
        return None
    quats  = pose[["quat_w","quat_x","quat_y","quat_z"]].values
    pitches = Rotation.from_quat(quats[:,[1,2,3,0]]).as_euler("xyz", degrees=True)[:,1]
    labels  = pd.Series([pitch_to_class(p) for p in pitches], index=pose.index)
    return labels, pose["frame"]


def load_session_tensor(session_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Returns (N_frames, N_channels, 12, 8) float32 and per-frame labels. All phases."""
    result = session_to_labels(session_path)
    if result is None:
        return None
    label_series, frame_series = result

    raw  = pd.read_csv(session_path / "tactile_raw.csv")
    pose = pd.read_csv(session_path / "object_pose.csv")[["frame"]]
    # Merge to align labels with raw frames
    pose["label"] = label_series.values
    raw  = raw.merge(pose, on="frame")
    if len(raw) < 5:
        return None

    raw_cols = [c for c in raw.columns if c not in ("frame","t_unix","t_rel","label")]
    channels = []

    for p in PATCHES_12x8:
        cols = patch_cols(raw_cols, p, PATCH_H, PATCH_W)
        arr  = raw[cols].fillna(0).values.astype(np.float32)
        channels.append(arr.reshape(-1, PATCH_H, PATCH_W))

    for p in PATCHES_10x8:
        cols = patch_cols(raw_cols, p, 10, PATCH_W)
        arr  = raw[cols].fillna(0).values.astype(np.float32)
        arr  = arr.reshape(-1, 10, PATCH_W)
        pad  = np.zeros((len(arr), 2, PATCH_W), dtype=np.float32)
        channels.append(np.concatenate([arr, pad], axis=1))

    X = np.stack(channels, axis=1)          # (N, 10, 12, 8)
    y = raw["label"].values.astype(np.int64)
    return X, y


# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading raw tactile data from sim sessions (all phases)...")
session_data = {}
for s in sorted(SIM_DIR.iterdir()):
    result = load_session_tensor(s)
    if result is not None:
        session_data[s.name] = result   # {name: (X, y_per_frame)}

print(f"  Loaded {len(session_data)} sessions")
all_labels = np.concatenate([y for _, y in session_data.values()])
for i, name in enumerate(CLASS_NAMES):
    n = (all_labels == i).sum()
    print(f"  [{i}] {name}: {n} frames ({n/len(all_labels)*100:.1f}%)")

# ── 80/20 session split ───────────────────────────────────────────────────────
sessions = np.array(sorted(session_data.keys()))
np.random.seed(42)
idx     = np.random.permutation(len(sessions))
n_train = int(len(sessions) * 0.8)
train_sessions = set(sessions[idx[:n_train]])
test_sessions  = set(sessions[idx[n_train:]])
print(f"  Train: {len(train_sessions)} sessions  |  Test: {len(test_sessions)} sessions")

def build_arrays(sess_set):
    Xs, ys = [], []
    for name in sess_set:
        X, y = session_data[name]
        Xs.append(X); ys.append(y)
    return np.concatenate(Xs), np.concatenate(ys)

X_train, y_train = build_arrays(train_sessions)
X_test,  y_test  = build_arrays(test_sessions)
print(f"  Train frames: {len(y_train)}  |  Test frames: {len(y_test)}")

# Global normalisation (fit on train)
mean = X_train.mean()
std  = X_train.std() + 1e-8
X_train = (X_train - mean) / std
X_test  = (X_test  - mean) / std


# ── Dataset ───────────────────────────────────────────────────────────────────
class TactileDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).long()
    def __len__(self):  return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


# ── Model ─────────────────────────────────────────────────────────────────────
class OrientationCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: (10, 12, 8) → (32, 6, 4)
            nn.Conv2d(N_CHANNELS, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),         nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            # Block 2: (32, 6, 4) → (64, 3, 2)
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            # Block 3: → (128, 1, 1)
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, N_CLASSES),
        )

    def forward(self, x): return self.classifier(self.features(x))


def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ── Training ──────────────────────────────────────────────────────────────────
EPOCHS     = 50
BATCH_SIZE = 128
LR         = 1e-3

ds_train = TactileDataset(X_train, y_train)
ds_test  = TactileDataset(X_test,  y_test)
dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,
                      num_workers=0, pin_memory=False)

model   = OrientationCNN().to(DEVICE)
opt     = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
loss_fn = nn.CrossEntropyLoss()

print(f"\nModel params: {count_params(model):,}")
print(f"Training {EPOCHS} epochs on {DEVICE}...")
print(f"{'Epoch':>6}  {'Loss':>8}  {'TrainAcc':>9}  {'TestAcc':>8}")

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for xb, yb in dl_train:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        logits = model(xb)
        loss   = loss_fn(logits, yb)
        loss.backward()
        opt.step()
        total_loss += loss.item() * len(yb)
        correct    += (logits.argmax(1) == yb).sum().item()
        total      += len(yb)
    sched.step()

    if epoch % 5 == 0 or epoch == 1:
        model.eval()
        with torch.no_grad():
            xv = ds_test.X.to(DEVICE)
            pv = model(xv).argmax(1).cpu().numpy()
        test_acc  = accuracy_score(y_test, pv)
        train_acc = correct / total
        print(f"{epoch:6d}  {total_loss/total:8.4f}  {train_acc:9.3f}  {test_acc:8.3f}")

# ── Final evaluation ──────────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    xv    = ds_test.X.to(DEVICE)
    preds = model(xv).argmax(1).cpu().numpy()
    probs = torch.softmax(model(xv), dim=1).cpu().numpy()

print("\n" + "="*50)
print("  CNN Orientation — Final Results")
print("="*50)
print(f"  Frame accuracy: {accuracy_score(y_test, preds):.3f}")
print(f"\n{classification_report(y_test, preds, target_names=['vertical','tilted','horizontal'])}")

# Session-level majority vote
print("  Per-session majority vote:")
sess_correct = []
for name in sorted(test_sessions):
    X_s, y_s = session_data[name]
    X_s_norm = torch.from_numpy((X_s - mean) / std).to(DEVICE)
    with torch.no_grad():
        p_s = model(X_s_norm).argmax(1).cpu().numpy()
    maj     = int(np.bincount(p_s, minlength=N_CLASSES).argmax())
    # Session true label = most common label in session
    true    = int(np.bincount(y_s, minlength=N_CLASSES).argmax())
    correct = maj == true
    sess_correct.append(correct)
    sym = "✓" if correct else "✗"
    print(f"    {sym} true={CLASS_NAMES[true][:10]:10s}  pred={CLASS_NAMES[maj][:10]}")

print(f"\n  Session accuracy: {np.mean(sess_correct):.3f}  ({sum(sess_correct)}/{len(sess_correct)})")

# ── Save ──────────────────────────────────────────────────────────────────────
torch.save({
    "model_state": model.state_dict(),
    "norm_mean":   float(mean),
    "norm_std":    float(std),
    "patches_12x8": PATCHES_12x8,
    "patches_10x8": PATCHES_10x8,
    "patch_h": PATCH_H, "patch_w": PATCH_W,
}, MODEL_DIR / "cnn_orientation.pt")
print(f"\nSaved → {MODEL_DIR}/cnn_orientation.pt")
