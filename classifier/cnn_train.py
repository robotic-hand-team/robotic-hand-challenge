"""
Small 2D-CNN for Sensor A vs B classification.

Input: 6 tactile patches (5 nail + thumb_pad), each 12x8 → tensor (6, 12, 8)
Architecture:
  Conv(6→16, 3x3) → BN → ReLU → MaxPool
  Conv(16→32, 3x3) → BN → ReLU → GlobalAvgPool
  Linear(32→2)
  ~7k parameters total

Evaluation: Leave-One-Session-Out CV (13 folds)
Then trains final model on all data and saves to model/cnn_sensor.pt
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score

HERE        = Path(__file__).parent
DATASET_DIR = HERE.parent / "models" / "dataset"
MODEL_DIR   = HERE / "model"
MODEL_DIR.mkdir(exist_ok=True)

# ── Sessions ──────────────────────────────────────────────────────────────────
SESSIONS_A = [
    "20261016_204344", "20261016_204402", "20261016_204429",
    "20261016_204457", "20261016_204513", "20261016_204524", "20261016_204548",
]
SESSIONS_B = [
    "20261016_204558", "20261016_204609", "20261016_204619",
    "20261016_204627", "20261016_204637", "20261016_204649",
]

# 12x8 patches used as CNN channels
PATCHES = ["little_nail", "ring_nail", "middle_nail", "index_nail", "thumb_nail", "thumb_pad"]
PATCH_H, PATCH_W = 12, 8

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ── Data loading ──────────────────────────────────────────────────────────────
def patch_columns(raw_cols: list[str], patch: str) -> list[str]:
    cols = [(c, int(c.split("_r")[1].split("_c")[0]), int(c.split("_c")[1]))
            for c in raw_cols if c.startswith(patch + "_r")]
    cols.sort(key=lambda x: (x[1], x[2]))
    return [c[0] for c in cols]


def load_session_raw(session: str, label: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns X (N, 6, 12, 8) and y (N,)."""
    raw = pd.read_csv(DATASET_DIR / session / "tactile_raw.csv")
    raw_cols = [c for c in raw.columns if c not in ("frame", "t_unix", "t_rel")]

    channels = []
    for patch in PATCHES:
        cols = patch_columns(raw_cols, patch)
        assert len(cols) == PATCH_H * PATCH_W, f"{patch}: expected {PATCH_H*PATCH_W}, got {len(cols)}"
        arr = raw[cols].fillna(0).values.astype(np.float32)  # (N, 96)
        arr = arr.reshape(-1, PATCH_H, PATCH_W)               # (N, 12, 8)
        channels.append(arr)

    X = np.stack(channels, axis=1)   # (N, 6, 12, 8)
    y = np.full(len(raw), label, dtype=np.int64)
    return X, y


class TactileDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, normalize: bool = True, stats=None):
        if normalize:
            if stats is None:
                self.mean = X.mean()
                self.std  = X.std() + 1e-8
            else:
                self.mean, self.std = stats
            X = (X - self.mean) / self.std
        else:
            self.mean, self.std = 0.0, 1.0
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):  return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


# ── Model ─────────────────────────────────────────────────────────────────────
class TactileCNN(nn.Module):
    def __init__(self, n_channels: int = 6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(n_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),                   # → (16, 6, 4)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),           # → (32, 1, 1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── Training ──────────────────────────────────────────────────────────────────
def train_model(X_train, y_train, X_val, y_val, epochs=25, lr=1e-3, batch_size=64):
    ds_train = TactileDataset(X_train, y_train)
    ds_val   = TactileDataset(X_val,   y_val, stats=(ds_train.mean, ds_train.std))
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)

    model = TactileCNN().to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for xb, yb in dl_train:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        xv = ds_val.X.to(DEVICE)
        preds = model(xv).argmax(1).cpu().numpy()
    acc = accuracy_score(y_val, preds)
    return model, acc, (ds_train.mean, ds_train.std)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Device: {DEVICE}")

    # Load all sessions
    print("Loading raw tactile data...")
    all_X, all_y, all_sess = [], [], []
    for s in SESSIONS_A:
        X, y = load_session_raw(s, 0)
        all_X.append(X); all_y.append(y)
        all_sess.extend([s] * len(y))
        print(f"  [A] {s}  {len(y)} frames")
    for s in SESSIONS_B:
        X, y = load_session_raw(s, 1)
        all_X.append(X); all_y.append(y)
        all_sess.extend([s] * len(y))
        print(f"  [B] {s}  {len(y)} frames")

    all_X    = np.concatenate(all_X)
    all_y    = np.concatenate(all_y)
    all_sess = np.array(all_sess)
    all_sess_list = SESSIONS_A + SESSIONS_B

    print(f"\nTotal: {len(all_y)} frames  |  model params: {count_params(TactileCNN()):,}")
    print(f"Input shape: {all_X.shape[1:]}")

    # Leave-one-session-out CV
    print(f"\nLeave-One-Session-Out CV ({len(all_sess_list)} folds)...")
    cv_accs = []
    for test_sess in all_sess_list:
        mask_test  = all_sess == test_sess
        mask_train = ~mask_test
        _, acc, _ = train_model(all_X[mask_train], all_y[mask_train],
                                all_X[mask_test],  all_y[mask_test])
        label = "A" if test_sess in SESSIONS_A else "B"
        print(f"  [{label}] {test_sess}  acc={acc:.3f}")
        cv_accs.append(acc)

    print(f"\n{'='*45}")
    print(f"  CNN LOSO-CV  mean={np.mean(cv_accs):.3f}  "
          f"std={np.std(cv_accs):.3f}  "
          f"min={np.min(cv_accs):.3f}  "
          f"max={np.max(cv_accs):.3f}")
    print(f"{'='*45}")

    # Final model on all data
    print("\nTraining final model on all data...")
    final_model, _, norm_stats = train_model(all_X, all_y, all_X, all_y, epochs=30)

    torch.save({
        "model_state": final_model.state_dict(),
        "norm_mean":   norm_stats[0],
        "norm_std":    norm_stats[1],
        "patches":     PATCHES,
        "patch_shape": (PATCH_H, PATCH_W),
    }, MODEL_DIR / "cnn_sensor.pt")

    print(f"Saved → {MODEL_DIR}/cnn_sensor.pt")
