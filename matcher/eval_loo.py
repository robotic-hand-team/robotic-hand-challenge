"""Leave-one-out accuracy evaluation of the DTW session matcher."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from matcher.match_session import extract_series, dtw_distance

LIB    = Path.home() / "Documents/20261017_17October_8pm"
LABELS = Path.home() / "Documents/labelled_data/manual_labels.csv"

labels_df   = pd.read_csv(LABELS)
sessions_map = {s.name.split("_", 1)[1]: s for s in LIB.iterdir() if s.is_dir()}

labeled = []
for _, row in labels_df.iterrows():
    ts  = row["filename"].replace("20261017_","").replace(".jpg","").replace("_error","")
    obj = row["object"]
    if ts in sessions_map and obj in ("sensor_A", "sensor_B"):
        res = extract_series(sessions_map[ts])
        if res is not None:
            labeled.append({"path": sessions_map[ts], "label": obj, "ts": res[0]})

print(f"Labeled sessions: {len(labeled)}  "
      f"({sum(x['label']=='sensor_A' for x in labeled)}× sensor_A, "
      f"{sum(x['label']=='sensor_B' for x in labeled)}× sensor_B)\n")

correct = 0
results = []
for i, query in enumerate(labeled):
    others  = [x for j, x in enumerate(labeled) if j != i]
    lib_ts  = [x["ts"]    for x in others]
    lib_lbl = [x["label"] for x in others]

    with ThreadPoolExecutor(max_workers=8) as ex:
        dists = list(ex.map(lambda b: dtw_distance(query["ts"], b), lib_ts))

    best      = int(np.argmin(dists))
    pred      = lib_lbl[best]
    hit       = pred == query["label"]
    correct  += hit
    results.append({"query": query["path"].name, "true": query["label"],
                    "pred": pred, "dist": dists[best], "correct": hit})
    print(f"[{'✓' if hit else '✗'}]  {query['path'].name:30s}  "
          f"true={query['label']:8s}  pred={pred:8s}  dist={dists[best]:.2f}")

acc = correct / len(labeled)
df  = pd.DataFrame(results)
print(f"\n{'='*62}")
print(f"  LOO Accuracy : {correct}/{len(labeled)} = {acc*100:.1f}%")
for cls in ("sensor_A", "sensor_B"):
    sub = df[df["true"] == cls]
    c   = int(sub["correct"].sum())
    print(f"  {cls}  : {c}/{len(sub)} = {c/len(sub)*100:.0f}%")
