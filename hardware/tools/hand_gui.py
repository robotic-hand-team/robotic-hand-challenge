import tkinter as tk
from pymodbus.client import ModbusTcpClient
import numpy as np
from PIL import Image, ImageTk
import os
import json
import time
import joblib
from pathlib import Path

# ----------------------------
# Verbindung
# ----------------------------
IP = "192.168.123.211"
PORT = 6000

client = ModbusTcpClient(IP, port=PORT)
client.connect()

# ----------------------------
# Datei für Speicher
# ----------------------------
SAVE_FILE = "positions.json"

# ----------------------------
# Klassifizierer für Sensor A/B
# ----------------------------
# Der XGBoost-Klassifizierer aus classifier/ schätzt aus den Taktildaten + Ist-Winkeln,
# welcher Sensor (A = VEGABAR 29, B = VEGABAR 38) gerade gehalten wird.
# Modell-Variante per Umgebungsvariable wählbar:
#   CLASSIFIER_SUFFIX=""          -> auf echten Hardware-Daten trainiert (Standard)
#   CLASSIFIER_SUFFIX="_combined" -> real + Simulation gemischt
#   CLASSIFIER_SUFFIX="_sim"      -> nur Simulation
CLASSIFIER_DIR = Path(__file__).parent.parent.parent / "classifier" / "model"
CLASSIFIER_SUFFIX = os.environ.get("CLASSIFIER_SUFFIX", "")
CLASSIFIER_LABEL_NAMES = {0: "Sensor A (VEGA 29)", 1: "Sensor B (VEGA 38)"}

CLASSIFIER_MODEL = None
CLASSIFIER_SCALER = None
CLASSIFIER_FEATURE_COLS = None

try:
    CLASSIFIER_MODEL = joblib.load(CLASSIFIER_DIR / f"xgb_sensor{CLASSIFIER_SUFFIX}.pkl")
    CLASSIFIER_SCALER = joblib.load(CLASSIFIER_DIR / f"scaler{CLASSIFIER_SUFFIX}.pkl")
    CLASSIFIER_FEATURE_COLS = joblib.load(CLASSIFIER_DIR / f"feature_cols{CLASSIFIER_SUFFIX}.pkl")
    print(f"Klassifizierer geladen: xgb_sensor{CLASSIFIER_SUFFIX}.pkl")
except Exception as e:
    print(f"Klassifizierer konnte nicht geladen werden: {e}")
    CLASSIFIER_MODEL = None

# ----------------------------
# GUI
# ----------------------------
root = tk.Tk()
root.title("Greifhand – mit Positionsspeicher")
root.geometry("1200x900")

left_frame = tk.Frame(root)
left_frame.pack(side="left", fill="y", padx=10)

right_frame = tk.Frame(root)
right_frame.pack(side="right", fill="both", expand=True)

# ----------------------------
# Bild (aus frontend/public/)
# ----------------------------
img_path = os.path.join(
    os.path.dirname(__file__), "..", "..", "frontend", "public", "hand.png"
)
img = Image.open(img_path)
img = img.resize((700, 900))
hand_img = ImageTk.PhotoImage(img)

# ----------------------------
# SENSOR LAYOUT
# ----------------------------
sensor_groups = [
    ("Little Tip", 3000, 3, 3, "Little", (8, 10)),
    ("Little Nail", 3018, 12, 8, "Little", (-2, 40)),
    ("Little Pad", 3210, 10, 8, "Little", (4, 190)),
    ("Ring Tip", 3370, 3, 3, "Ring", (-9, -20)),
    ("Ring Nail", 3388, 12, 8, "Ring", (-23, 30)),
    ("Ring Pad", 3580, 10, 8, "Ring", (-18, 200)),
    ("Middle Tip", 3740, 3, 3, "Middle", (-21, -25)),
    ("Middle Nail", 3758, 12, 8, "Middle", (-39, 22)),
    ("Middle Pad", 3950, 10, 8, "Middle", (-35, 210)),
    ("Index Tip", 4110, 3, 3, "Index", (-38, -19)),
    ("Index Nail", 4128, 12, 8, "Index", (-49, 17)),
    ("Index Pad", 4320, 10, 8, "Index", (-54, 200)),
    ("Thumb Tip", 4480, 3, 3, "Thumb", (185, 140)),
    ("Thumb Nail", 4498, 12, 8, "Thumb", (145, 170)),
    ("Thumb Mid", 4690, 3, 3, "Thumb", (90, 240)),
    ("Thumb Pad", 4708, 12, 8, "Thumb", (-30, 240)),
    ("Palm", 4900, 8, 14, "Palm", (-15, 0)),
]

offsets = {
    "Little": (150, 120),
    "Ring": (230, 110),
    "Middle": (310, 100),
    "Index": (390, 110),
    "Thumb": (420, 360),
    "Palm": (240, 520),
}

DOF_NAMES = ["little", "ring", "middle", "index", "thumb_bend", "thumb_rot"]

pixel = 6
sensor_offsets = {}

# Zustand der Sensor-Schätzung
prediction_votes = []      # Labels der aktiven Frames für die Mehrheitsentscheidung
last_prediction = None     # letztes Vorhersage-Ergebnis (dict) oder None


# ----------------------------
# MODBUS
# ----------------------------
def safe_read(addr, length):
    values = []
    for i in range(0, length, 120):
        res = client.read_holding_registers(
            address=addr + i, count=min(120, length - i), device_id=1
        )
        if not res.isError():
            values.extend(res.registers)
    return values


# ----------------------------
# OFFSET KALIBRIERUNG
# ----------------------------
def calibrate_offsets():
    print("Kalibriere Sensoren...")
    time.sleep(1)

    for name, addr, h, w, key, rel in sensor_groups:
        data = safe_read(addr, h * w)
        sensor_offsets[name] = np.array(data) if data else np.zeros(h * w)


# ----------------------------
# STEUERUNG
# ----------------------------
def send_angles():
    client.write_registers(address=1486, values=[s.get() for s in sliders], device_id=1)


def open_hand():
    for s in sliders:
        s.set(1000)
    send_angles()


def close_hand():
    vals = [0, 0, 0, 0, 234, 670]
    for i in range(6):
        sliders[i].set(vals[i])
    send_angles()


def step_plus():
    for s in sliders:
        s.set(min(1000, s.get() + 20))
    send_angles()


def step_minus():
    for s in sliders:
        s.set(max(0, s.get() - 20))
    send_angles()


# ----------------------------
# SENSOR-DATEN AUFBEREITEN
# ----------------------------
def read_tactile_matrices():
    """Liest alle Taktilgruppen und liefert baseline-korrigierte, geclippte Matrizen."""
    matrices = {}
    for name, addr, h, w, key, rel in sensor_groups:
        data = safe_read(addr, h * w)
        if not data:
            continue
        raw = np.array(data)
        offset = sensor_offsets.get(name, np.zeros(len(raw)))
        matrix = (raw - offset).reshape(h, w)
        matrix[matrix < 0] = 0
        matrices[name] = matrix
    return matrices


def tactile_summary_from(matrices):
    """Zusammenfassung je Gruppe – gleiche Logik wie die Aufnahme/Datensatz-Summary."""
    summary = {}
    for name, addr, h, w, key, rel in sensor_groups:
        key_name = name.lower().replace(" ", "_")
        matrix = matrices.get(name)
        if matrix is None:
            summary[key_name] = {"sum": 0, "max": 0, "contacts": 0,
                                 "cy": 0.0, "cx": 0.0, "force": 0.0}
            continue

        active = matrix > 50
        n_contacts = int(np.sum(active))
        entry = {"sum": int(matrix.sum()), "max": int(matrix.max()),
                 "contacts": n_contacts}
        if n_contacts >= 2:
            y_idx, x_idx = np.where(active)
            entry["cy"] = float(np.mean(y_idx))
            entry["cx"] = float(np.mean(x_idx))
            entry["force"] = float(np.mean(matrix[active]))
        else:
            entry["cy"] = 0.0
            entry["cx"] = 0.0
            entry["force"] = 0.0
        summary[key_name] = entry
    return summary


# ----------------------------
# KLASSIFIZIERER-VORHERSAGE
# ----------------------------
def predict_sensor(summary, angles):
    """Schätzt aus Taktil-Zusammenfassung + Ist-Winkeln den gehaltenen Sensor."""
    global last_prediction

    if CLASSIFIER_MODEL is None:
        return None

    row = {}
    for key, s in summary.items():
        for stat in ("sum", "max", "contacts", "cy", "cx", "force"):
            row[f"{key}_{stat}"] = float(s[stat])
    for n, v in zip(DOF_NAMES, angles):
        row[f"angle_act_{n}"] = float(v)

    try:
        x = np.array([[row.get(c, 0.0) for c in CLASSIFIER_FEATURE_COLS]])
        x_scaled = CLASSIFIER_SCALER.transform(x)
        label = int(CLASSIFIER_MODEL.predict(x_scaled)[0])
        confidence = float(CLASSIFIER_MODEL.predict_proba(x_scaled)[0].max())
    except Exception as e:
        print(f"Vorhersagefehler: {e}")
        return None

    total_contacts = sum(s["contacts"] for s in summary.values())
    active = total_contacts > 20          # genug Kontakt für eine sinnvolle Aussage
    if active:
        prediction_votes.append(label)

    last_prediction = {
        "label": label,
        "name": CLASSIFIER_LABEL_NAMES[label],
        "confidence": confidence,
        "active": active,
        "contacts": total_contacts,
    }
    return last_prediction


def update_prediction_display():
    """Aktualisiert die Vorhersage-Box (Mehrheitsentscheidung über die letzten Frames)."""
    pred = last_prediction

    if CLASSIFIER_MODEL is None:
        text, color = "Klassifizierer:\n(nicht geladen)", "#dddddd"
    elif pred is None:
        text, color = "Klassifizierer:\n(warten...)", "#dddddd"
    elif not pred["active"]:
        text, color = f"Klassifizierer:\n(warten auf Kontakt)\n{pred['name']}", "#ffeeaa"
    else:
        window = prediction_votes[-10:]
        if len(window) >= 5:
            majority = int(np.bincount(window).argmax())
            agreement = window.count(majority) / len(window)
            text = f"Klassifizierer:\n{CLASSIFIER_LABEL_NAMES[majority]}\n{agreement * 100:.0f}% Sicher"
        else:
            text = f"Klassifizierer:\n{pred['name']}\n{pred['confidence'] * 100:.0f}%"
        color = "#aaffaa"

    prediction_text.config(text=text, bg=color)


def reset_prediction():
    prediction_votes.clear()
    global last_prediction
    last_prediction = None
    update_prediction_display()


# ----------------------------
# 💾 SPEICHERFUNKTION
# ----------------------------
def save_position():
    name = entry.get()
    if not name:
        return

    values = [s.get() for s in sliders]
    positions[name] = values
    save_to_file()
    create_button(name, values)


def create_button(name, values):
    def apply():
        for i in range(6):
            sliders[i].set(values[i])
        send_angles()

    tk.Button(saved_frame, text=name, command=apply).pack(fill="x")


def save_to_file():
    with open(SAVE_FILE, "w", encoding="utf-8") as f:
        json.dump(positions, f)


def load_positions():
    if not os.path.exists(SAVE_FILE):
        return

    with open(SAVE_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    for name, values in data.items():
        positions[name] = values
        create_button(name, values)


# ----------------------------
# VISUALISIERUNG
# ----------------------------
def draw(matrices):
    canvas.delete("all")
    canvas.create_image(350, 450, image=hand_img)

    for name, addr, h, w, key, rel in sensor_groups:
        matrix = matrices.get(name)
        if matrix is None:
            continue

        x0 = offsets[key][0] + rel[0]
        y0 = offsets[key][1] + rel[1]

        # Heatmap
        for i in range(h):
            for j in range(w):
                v = matrix[i, j] / 4000

                if v < 0.2:
                    color = "#00aa00"
                elif v < 0.5:
                    color = "#cccc00"
                elif v < 0.8:
                    color = "#ff8800"
                else:
                    color = "#ff0000"

                canvas.create_rectangle(
                    x0 + j * pixel,
                    y0 + i * pixel,
                    x0 + (j + 1) * pixel,
                    y0 + (i + 1) * pixel,
                    fill=color,
                    width=0,
                    stipple="gray25",
                )

        active = matrix > 50
        if np.sum(active) < 2:
            continue

        y_idx, x_idx = np.where(active)
        cx = int(np.mean(x_idx))
        cy = int(np.mean(y_idx))
        force = np.mean(matrix[active])
        size = int(3 + force / 200)

        canvas.create_oval(
            x0 + cx * pixel,
            y0 + cy * pixel,
            x0 + cx * pixel + size,
            y0 + cy * pixel + size,
            fill="blue",
        )


# ----------------------------
# UI
# ----------------------------
sliders = []
positions = {}

names = [
    "Kleiner Finger",
    "Ringfinger",
    "Mittelfinger",
    "Zeigefinger",
    "Daumen B",
    "Daumen R",
]

for i in range(6):
    f = tk.Frame(left_frame)
    f.pack()

    tk.Label(f, text=names[i], width=15).pack(side="left")

    s = tk.Scale(f, from_=0, to=1000, orient="horizontal", length=250)
    s.pack(side="left")
    s.bind("<ButtonRelease-1>", lambda e: send_angles())

    sliders.append(s)


def read_current_angles():
    res = client.read_holding_registers(address=1486, count=6, device_id=1)
    if not res.isError():
        for i in range(6):
            sliders[i].set(res.registers[i])


root.after(500, read_current_angles)
root.after(1000, calibrate_offsets)

# Buttons
bf = tk.Frame(left_frame)
bf.pack(pady=10)

tk.Button(bf, text="Öffnen", command=open_hand).pack(side="left")
tk.Button(bf, text="Schließen", command=close_hand).pack(side="left")
tk.Button(bf, text="Step +", command=step_plus).pack(side="left")
tk.Button(bf, text="Step -", command=step_minus).pack(side="left")

# ----------------------------
# Klassifizierer-Box
# ----------------------------
tk.Label(left_frame, text="Sensor-Schätzung (ML)", font=("Arial", 10, "bold")).pack(pady=(20, 2))

prediction_text = tk.Label(
    left_frame,
    text="Klassifizierer:\n(warten...)",
    font=("Arial", 13, "bold"),
    bg="#dddddd",
    fg="#000000",
    width=22,
    height=4,
    relief="ridge",
    borderwidth=2,
    justify="center",
)
prediction_text.pack(pady=5)

tk.Button(left_frame, text="Schätzung zurücksetzen", command=reset_prediction).pack(fill="x")

# ----------------------------
# 💾 Speicher UI
# ----------------------------
entry = tk.Entry(left_frame)
entry.pack(fill="x", pady=5)

tk.Button(left_frame, text="Position speichern", command=save_position).pack(fill="x")

saved_frame = tk.Frame(left_frame)
saved_frame.pack(fill="both", expand=True)

load_positions()

# ----------------------------
# Canvas
# ----------------------------
canvas = tk.Canvas(right_frame, width=700, height=900)
canvas.pack()


# ----------------------------
# HAUPTLOOP
# ----------------------------
def loop():
    matrices = read_tactile_matrices()
    draw(matrices)

    summary = tactile_summary_from(matrices)
    res = client.read_holding_registers(address=1546, count=6, device_id=1)  # angle_act
    angles = res.registers if not res.isError() else [0] * 6
    predict_sensor(summary, angles)
    update_prediction_display()

    root.after(120, loop)


loop()
root.mainloop()
