import tkinter as tk
from pymodbus.client import ModbusTcpClient
import numpy as np
from PIL import Image, ImageTk
import os
import csv
import json
import time
import signal
import threading
from datetime import datetime

# ----------------------------
# Connection
# ----------------------------
IP = "192.168.123.210"
PORT = 6000

client = ModbusTcpClient(IP, port=PORT)
client.connect()

# pymodbus is not thread-safe: every request goes through this lock
modbus_lock = threading.Lock()

# ----------------------------
# File for saved positions
# ----------------------------
SAVE_FILE = "positions.json"

# ----------------------------
# ACQUISITION / LOGGING
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")

SAMPLE_RATE_HZ = 50        # target rate of the acquisition thread (capped by Modbus speed)
GUI_RATE_MS = 100          # heatmap redraw interval (independent of logging)
LOG_AUTOSTART = False      # True = start logging automatically after calibration
LOG_RAW_TACTILE = True     # write the full tactile matrices for every frame
CALIBRATION_SAMPLES = 10   # samples averaged for the tactile zero offset

# Registers from the RH56DFTP user manual (section 2.6).
# Addresses are BYTE addresses; one Modbus register = 2 bytes.
ACT_BLOCK_START = 1474     # POS_SET ... TEMP read in a single request
ACT_BLOCK_WORDS = 75       # bytes 1474–1623

ACT_SHORT_FIELDS = {       # name: (address, signed)  – 6 short values, one per DOF
    "pos_set":   (1474, False),
    "angle_set": (1486, False),
    "force_set": (1498, False),
    "speed_set": (1522, False),
    "pos_act":   (1534, False),
    "angle_act": (1546, False),
    "force_act": (1582, True),
    "current":   (1594, True),
}
ACT_BYTE_FIELDS = {        # name: address  – 6 byte values, one per DOF
    "error":  1606,
    "status": 1612,
    "temp":   1618,
}
ACT_FIELDS = list(ACT_SHORT_FIELDS) + list(ACT_BYTE_FIELDS)

CLEAR_ERROR_ADDR = 1004    # write 1 to clear the latched error flags of all DOFs

# ERROR / STATUS byte meanings (RH56 manual, section 2.6)
ERROR_BITS = {0: "stall", 1: "over-temp", 2: "over-current", 3: "motor fault", 4: "comm error"}
# the manual gives no meaning for STATUS; inferred from the recorded dataset (see AGENTS.md)
STATUS_NAMES = {0: "releasing", 1: "closing/hold", 2: "force stop", 255: "not init"}

FORCE_LIMIT = 3000          # force_set per DOF (0–3000); the finger stops gently at this force
FORCE_LIMIT_THUMB_BEND = 1000  # force_set of thumb bending
FORCE_LIMIT_THUMB_ROT = 200    # force_set of thumb rotation: low, so it stops softly on the object
                               # (its force reading at rest has been ~50, but also ~440 – then it can't move)
OPEN_FORCE_LIMIT = 3000    # force_set while opening: releasing squeezes nothing (hand default)
SPEED_FINGERS = 1000       # speed_set of the fingers (0–1000; 1000 = full stroke in ~0.6 s)
SPEED_THUMB = 1000          # slower thumb: gentler contact, lower current, time for the guard to react

# Thumb rotation guard: the hand's force limit never stops a rotation that pushes the object
# sideways (its force reading goes negative), so the software stops it when any of these trips
ROT_STOP_PRESSURE = 1000    # thumb taxel value above baseline (on >= CONTACT_MIN_TAXELS taxels)
ROT_STOP_FORCE = 200       # |thumb_rot force - its resting value|, either sign
ROT_STOP_CURRENT = 400     # thumb_rot current in mA
ROT_STEP = 50              # rotation sweeps move in steps of this size, one per STAGE_DELAY_MS
STAGE_DELAY_MS = 300       # pause between the stages of open/close so thumb and fingers don't collide
TEMP_WARN_C = 60           # highlight the status readout above this motor temperature
TEMP_VALID_MAX = 100       # manual: TEMP range is 0–100 °C, above is a corrupted reading (display only)

# Hand state machine
# a DOF at or above its angle counts as fully open (order of DOF_KEYS);
# thumb_rot is lower because it currently stops at ~817 instead of 1000
OPEN_ANGLES = [950, 950, 950, 950, 950, 750]
STILL_TOL = 3              # angle change per GUI tick below which a DOF counts as not moving
STILL_S = 0.4              # the hand must be still this long for a move to count as finished
NO_OBJECT_S = 0.5          # no pad/palm contact this long in HOLD -> "No object" shown
HOLD_STOP_S = 1.0          # this long after entering HOLD all motors stop (targets = measured angles)
MOVE_TIMEOUT_S = 3.0       # opening/closing that takes longer than this -> warning (not FAULTY)
ERROR_CLEAR_GRACE_S = 0.5  # ignore error flags this long after clearing them

TACTILE_START = 3000       # all tactile areas are one contiguous block
TACTILE_WORDS = 1062       # bytes 3000–5123

DOF_KEYS = ["little", "ring", "middle", "index", "thumb_bend", "thumb_rot"]

# ----------------------------
# GUI
# ----------------------------
root = tk.Tk()
root.title("Gripper Hand – with Position Memory")
root.geometry("1200x900")

left_frame = tk.Frame(root)
left_frame.pack(side="left", fill="y", padx=10)

right_frame = tk.Frame(root)
right_frame.pack(side="right", fill="both", expand=True)

# ----------------------------
# Image
# ----------------------------
img = Image.open(os.path.join(BASE_DIR, "hand.png"))
img = img.resize((700, 900))
hand_img = ImageTk.PhotoImage(img)

# ----------------------------
# SENSOR LAYOUT (UNCHANGED)
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

    ("Palm", 4900, 8, 14, "Palm", (-15, 0))
]

offsets = {
    "Little": (150, 120),
    "Ring":   (230, 110),
    "Middle": (310, 100),
    "Index":  (390, 110),
    "Thumb":  (420, 360),
    "Palm":   (240, 520)
}

pixel = 6
sensor_offsets = {}
calibrated = False

CONTACT_THRESHOLD = 30

# Auto hold: close the hand when an object is placed on the palm
CONTACT_MIN_TAXELS = 5       # taxels above CONTACT_THRESHOLD that count as a contact
AUTO_HOLD_TRIGGER_S = 0.1    # contact must persist this long before the hand closes
AUTO_HOLD_REARM_S = 1.0      # palm must stay clear this long before auto hold can trigger again

# ----------------------------
# MODBUS
# ----------------------------
def safe_read(addr, length):
    """Reads `length` registers in chunks of 120. Returns [] if any chunk fails."""
    values = []
    with modbus_lock:
        for i in range(0, length, 120):
            res = client.read_holding_registers(
                address=addr+i*2,          # byte address advances 2 per register
                count=min(120, length-i),
                device_id=1
            )
            if res.isError():
                return []
            values.extend(res.registers)
    return values

def to_signed(v):
    return v - 65536 if v > 32767 else v

def read_actuators():
    """One request for all actuator registers. Returns {field: [6 values]} or None."""
    words = safe_read(ACT_BLOCK_START, ACT_BLOCK_WORDS)
    if len(words) != ACT_BLOCK_WORDS:
        return None
    out = {}
    for name, (addr, signed) in ACT_SHORT_FIELDS.items():
        i = (addr - ACT_BLOCK_START) // 2
        vals = words[i:i+6]
        out[name] = [to_signed(v) for v in vals] if signed else list(vals)
    for name, addr in ACT_BYTE_FIELDS.items():
        i = (addr - ACT_BLOCK_START) // 2
        vals = []
        for w in words[i:i+3]:              # low byte = first DOF
            vals.append(w & 0xFF)
            vals.append((w >> 8) & 0xFF)
        out[name] = vals
    return out

def read_tactile_raw():
    """One block read of all taxels. Returns {name: flat raw array} or None."""
    words = safe_read(TACTILE_START, TACTILE_WORDS)
    if len(words) != TACTILE_WORDS:
        return None
    words = np.array(words)
    raw = {}
    for name, addr, h, w, key, rel in sensor_groups:
        i = (addr - TACTILE_START) // 2
        raw[name] = words[i:i+h*w]
    return raw

def correct_tactile(raw):
    """Subtracts calibration offsets, returns {name: matrix >= 0}."""
    frames = {}
    for name, addr, h, w, key, rel in sensor_groups:
        m = (raw[name] - sensor_offsets.get(name, 0)).reshape(h, w)
        m[m < 0] = 0
        frames[name] = m
    return frames

def contact_info(matrix):
    """Contact count, centroid (row/col) and mean force."""
    active = matrix > CONTACT_THRESHOLD
    n = int(np.sum(active))
    if n < 2:
        return n, None, None, 0.0
    y_idx, x_idx = np.where(active)
    return n, float(np.mean(y_idx)), float(np.mean(x_idx)), float(np.mean(matrix[active]))

# ----------------------------
# OFFSET CALIBRATION
# ----------------------------
def calibrate_offsets():
    global calibrated
    print("Calibrating sensors...")
    samples = []
    while len(samples) < CALIBRATION_SAMPLES and not stop_event.is_set():
        raw = read_tactile_raw()
        if raw is not None:
            samples.append(raw)
        time.sleep(0.02)
    for name, *_ in sensor_groups:
        sensor_offsets[name] = np.mean([s[name] for s in samples], axis=0) if samples else 0
    calibrated = True
    print(f"Calibration done ({len(samples)} samples).")
    check_block_mapping()

def check_block_mapping():
    """Sanity check: the palm read on its own must match its slice of the block read."""
    name, addr, h, w = "Palm", 4900, 8, 14
    raw = read_tactile_raw()
    direct = safe_read(addr, h*w)
    if raw is None or len(direct) != h*w:
        return
    diff = np.mean(np.abs(raw[name] - np.array(direct)))
    scale = max(1.0, np.mean(np.abs(direct)))
    if diff > 0.2 * scale + 20:
        print(f"WARNING: block read does not match direct read of {name} "
              f"(mean diff {diff:.0f}). Tactile data may be mis-mapped!")
    else:
        print(f"Block read check OK (mean diff {diff:.1f}).")

# ----------------------------
# LOGGER
# ----------------------------
class DataLogger:
    """
    Writes one folder per session, logs/YYYYmmdd_HHMMSS/, containing:
      actuators.csv        – position/angle/force/speed targets and actuals, current,
                             error, status, temperature per DOF
      tactile_summary.csv  – per sensor group: sum, max, contact points, centroid, mean force
      tactile_raw.csv      – every taxel (offset-corrected), column = "<group>_r<row>_c<col>"
      meta.json            – sensor layout, calibration offsets, start time
    All files share the columns 'frame', 't_unix', 't_rel'.
    start()/stop() are called from the GUI thread, log() from the acquisition thread.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.active = False
        self.files = []
        self.frame = 0
        self.t0 = None
        self.session_dir = None

    def start(self):
        with self.lock:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.session_dir = os.path.join(LOG_DIR, stamp)
            os.makedirs(self.session_dir, exist_ok=True)
            self.t0 = time.time()
            self.frame = 0

            meta = {
                "start": datetime.now().isoformat(),
                "ip": IP,
                "dof": DOF_KEYS,
                "sample_rate_target_hz": SAMPLE_RATE_HZ,
                "contact_threshold": CONTACT_THRESHOLD,
                "calibrated": calibrated,
                "sensor_groups": [
                    {"name": n, "addr": a, "rows": h, "cols": w, "finger": k}
                    for n, a, h, w, k, _ in sensor_groups
                ],
                "offsets": {n: np.round(np.asarray(o), 2).tolist() for n, o in sensor_offsets.items()},
            }
            with open(os.path.join(self.session_dir, "meta.json"), "w") as f:
                json.dump(meta, f, indent=2)

            base = ["frame", "t_unix", "t_rel"]

            act_cols = base[:]
            for field in ACT_FIELDS:
                act_cols += [f"{field}_{d}" for d in DOF_KEYS]

            sum_cols = base[:]
            for name, *_ in sensor_groups:
                k = name.lower().replace(" ", "_")
                sum_cols += [f"{k}_sum", f"{k}_max", f"{k}_contacts",
                             f"{k}_cy", f"{k}_cx", f"{k}_force"]

            self.act_w = self._open("actuators.csv", act_cols)
            self.sum_w = self._open("tactile_summary.csv", sum_cols)

            if LOG_RAW_TACTILE:
                raw_cols = base[:]
                for name, addr, h, w, *_ in sensor_groups:
                    k = name.lower().replace(" ", "_")
                    raw_cols += [f"{k}_r{i}_c{j}" for i in range(h) for j in range(w)]
                self.raw_w = self._open("tactile_raw.csv", raw_cols)

            self.active = True
        print(f"Logging started: {self.session_dir}")

    def _open(self, fname, header):
        f = open(os.path.join(self.session_dir, fname), "w", newline="")
        w = csv.writer(f)
        w.writerow(header)
        self.files.append(f)
        return w

    def log(self, t, actuators, tactile):
        with self.lock:
            if not self.active:
                return
            if self.frame == 0:
                self.t0 = t          # t_rel = 0 at the first logged sample
            base =[self.frame, f"{t:.4f}", f"{t - self.t0:.4f}"]

            row = base[:]
            for field in ACT_FIELDS:
                row += actuators[field]
            self.act_w.writerow(row)

            row = base[:]
            for name, *_ in sensor_groups:
                m = tactile[name]
                n, cy, cx, force = contact_info(m)
                row += [int(m.sum()), int(m.max()), n,
                        "" if cy is None else f"{cy:.2f}",
                        "" if cx is None else f"{cx:.2f}",
                        f"{force:.1f}"]
            self.sum_w.writerow(row)

            if LOG_RAW_TACTILE:
                row = base[:]
                for name, *_ in sensor_groups:
                    row += tactile[name].astype(int).ravel().tolist()
                self.raw_w.writerow(row)

            self.frame += 1
            if self.frame % 50 == 0:
                for f in self.files:
                    f.flush()

    def stop(self):
        with self.lock:
            if not self.active:
                return
            for f in self.files:
                f.close()
            self.files = []
            self.active = False
        print(f"Logging stopped ({self.frame} frames, "
              f"{self.frame / max(1e-6, time.time() - self.t0):.1f} Hz): {self.session_dir}")

logger = DataLogger()

# ----------------------------
# ACQUISITION THREAD
# ----------------------------
stop_event = threading.Event()
latest_lock = threading.Lock()
latest = {"tactile": None, "actuators": None, "t": 0.0}
stats = {"rate_hz": 0.0, "errors": 0}

def acquisition_loop():
    calibrate_offsets()
    if LOG_AUTOSTART:
        root.after(0, toggle_logging)

    period = 1.0 / SAMPLE_RATE_HZ
    next_t = time.perf_counter()
    last_t = None

    while not stop_event.is_set():
        t = time.time()
        raw = read_tactile_raw()
        act = read_actuators()

        if raw is None or act is None:
            stats["errors"] += 1
        else:
            tactile = correct_tactile(raw)
            with latest_lock:
                latest["tactile"] = tactile
                latest["actuators"] = act
                latest["t"] = t
            logger.log(t, act, tactile)
            rot_guard.update(act, tactile, hand["state"])

            if last_t is not None:   # smoothed actual rate for the GUI
                stats["rate_hz"] = 0.9 * stats["rate_hz"] + 0.1 / max(1e-6, t - last_t)
            last_t = t

        # fixed-rate schedule; if we fall behind, resync instead of bursting
        next_t += period
        delay = next_t - time.perf_counter()
        if delay > 0:
            stop_event.wait(delay)
        else:
            next_t = time.perf_counter()

acq_thread = threading.Thread(target=acquisition_loop, daemon=True)

def toggle_logging():
    if logger.active:
        logger.stop()
        log_button.config(text="● Start logging", bg=log_button_bg)
    else:
        if not calibrated:
            print("Note: sensors not calibrated yet – offsets are 0.")
        logger.start()
        log_button.config(text="■ Stop logging", bg="#ff6666")

def update_log_label():
    rate = f"Acquisition: {stats['rate_hz']:.1f} Hz (target {SAMPLE_RATE_HZ})"
    if stats["errors"]:
        rate += f", {stats['errors']} read errors"
    if logger.active:
        log_label.config(text=f"{rate}\n{os.path.relpath(logger.session_dir, BASE_DIR)}\n"
                              f"{logger.frame} frames, {time.time() - logger.t0:.1f} s")
    else:
        log_label.config(text=f"{rate}\nLogging off")

# ----------------------------
# CONTROL
# ----------------------------
def write_shorts(addr, values):
    """Writes registers; returns False (and prints why) if the write failed."""
    try:
        with modbus_lock:
            res = client.write_registers(address=addr, values=values, device_id=1)
    except Exception as e:      # pymodbus raises on connection problems
        print(f"Write to {addr} failed: {e}")
        return False
    if res.isError():
        print(f"Write to {addr} failed: {res}")
        return False
    return True

errors_cleared_at = 0.0

def clear_errors():
    global errors_cleared_at
    if write_shorts(CLEAR_ERROR_ADDR, [1]):
        errors_cleared_at = time.time()
        print("Cleared hand error flags.")

def set_force_limit(value=None):
    """Fingers: value (without one: the Grip force slider). Thumb: FORCE_LIMIT_THUMB_BEND and
    FORCE_LIMIT_THUMB_ROT."""
    write_shorts(ACT_SHORT_FIELDS["force_set"][0], grasp_limits(value))

def grasp_limits(value=None):
    value = int(force_slider.get() if value is None else value)
    return [value] * 4 + [FORCE_LIMIT_THUMB_BEND, FORCE_LIMIT_THUMB_ROT]

def set_move_limits(targets):
    """Manual moves, per DOF: closing gets the grasp limit, opening OPEN_FORCE_LIMIT
    (a low limit would stop a DOF whose force reading rises while it opens)."""
    with latest_lock:
        act = latest["actuators"]
    limits = grasp_limits()
    if act is not None:
        limits = [g if t < a else OPEN_FORCE_LIMIT
                  for t, a, g in zip(targets, act["angle_act"], limits)]
    write_shorts(ACT_SHORT_FIELDS["force_set"][0], limits)

def write_angles():
    # latched errors are never cleared here – only the Clear errors button does that
    targets = [s.get() for s in sliders]
    if rot_guard.reason and hand["state"] != "RELEASING":
        # thumb is pressing: keep the rotation where the guard stopped it (Open is exempt)
        targets[5] = rot_guard.stop_angle
        sliders[5].set(targets[5])
    write_shorts(ACT_SHORT_FIELDS["angle_set"][0], targets)

def hold_position(act):
    """Stop driving: every target becomes the measured angle, so the motors stop pushing
    (the lead screws keep the fingers where they are)."""
    cancel_pending_moves()
    for s, a in zip(sliders, act["angle_act"]):
        s.set(a)
    write_shorts(ACT_SHORT_FIELDS["angle_set"][0], list(act["angle_act"]))

def motion_blocked():
    """True (and the command is dropped) while the hand is FAULTY."""
    if hand["state"] != "FAULTY":
        return False
    print(f"Command ignored: hand is FAULTY ({hand['reason']}) – press Clear errors first.")
    with latest_lock:
        act = latest["actuators"]
    if act is not None:         # put the sliders back to what the hand is really set to
        for s, a in zip(sliders, act["angle_set"]):
            s.set(a)
    return True

# open/close move in stages; any new command cancels the stages still pending
pending_moves = []

def cancel_pending_moves():
    for after_id in pending_moves:
        root.after_cancel(after_id)
    pending_moves.clear()

def run_stages(stages):
    """stages: list of {dof_index: angle}, applied STAGE_DELAY_MS apart."""
    cancel_pending_moves()
    def apply(k):
        pending_moves.clear()       # this stage is running now
        for i, v in stages[k].items():
            sliders[i].set(v)
        write_angles()
        if k + 1 < len(stages):
            pending_moves.append(root.after(STAGE_DELAY_MS, apply, k + 1))
    apply(0)

def send_angles():
    """Manual command (slider, Step, saved position): the hand leaves its open/closed state."""
    if motion_blocked():
        return
    cancel_pending_moves()
    set_move_limits([s.get() for s in sliders])
    write_angles()
    set_hand_state("IDLE", "manual pose")

def open_hand():
    if motion_blocked():
        return
    # the grip force limit would stop a DOF whose force reading rises while opening
    write_shorts(ACT_SHORT_FIELDS["force_set"][0], [OPEN_FORCE_LIMIT] * 6)
    # thumb bend first so it lets go of the fingers, thumb rotation last
    run_stages([{4: 1000}, {0: 1000, 1: 1000, 2: 1000, 3: 1000}, {5: 1000}])
    set_hand_state("RELEASING")

def rotation_sweep(start, end, step=ROT_STEP):
    """Stages that move the thumb rotation from start to end in small steps, so the guard
    gets a check between steps instead of one big push."""
    sign = 1 if end > start else -1
    return [{5: a} for a in range(start + sign * step, end, sign * step)] + [{5: end}]

def close_hand():
    if motion_blocked():
        return
    set_force_limit()           # back to the grip force (Open raised it)
    # fingers first, then the thumb, so the thumb ends up over the fingers, not trapped under them
    run_stages([{5: 0, 4: 0},{0: 0, 1: 0},{5:110},{2: 0, 3: 600}, *rotation_sweep(110, 1000), {3:0}])
    set_hand_state("CLOSING")

def clear_faults():
    # if the hand is still too hot it sets the over-temp flag again and the state returns to FAULTY
    clear_errors()
    set_hand_state("IDLE", "errors cleared")

# ----------------------------
# 🤚 HAND STATE
# ----------------------------
# IDLE       startup, manual pose or after Clear errors; becomes OPENED once the hand is fully open
# OPENED     every DOF fully open – the only state in which auto close may fire
# CLOSING    Close sent, fingers moving
# CONTACT    a finger or the palm touched something while closing
# HOLD       closing finished (target reached or stopped on the object); stays until Open
# RELEASING  Open sent; becomes OPENED once every DOF is open
# FAULTY     an error register of a DOF is set (only the hand's flags, no software thresholds);
#            with the over-temp flag every DOF also stops driving. All motion commands are
#            refused; only Clear errors leaves it – faults are never cleared automatically.
#            A move that takes longer than MOVE_TIMEOUT_S only shows a warning.
# The transitions come from the hand's feedback (angles, error flags, tactile), not from
# the button presses alone.
HAND_STATE_COLORS = {"IDLE": "gray40", "OPENED": "#008800", "CLOSING": "#cc7700",
                     "CONTACT": "#0066cc", "HOLD": "#003399", "RELEASING": "#cc7700",
                     "FAULTY": "red"}

hand = {
    "state": "IDLE", "since": time.time(), "reason": "",
    "last_angles": None, "still_since": time.time(),    # motion detection
    "palm_clear_since": None, "contact_since": None,    # auto close timers
    "auto_armed": False,
    "motors_off": False,                                # HOLD: motors already stopped
    "no_contact_since": None,                           # "No object" display in HOLD
}

def set_hand_state(state, reason=""):
    if state != hand["state"]:
        print(f"Hand state: {hand['state']} -> {state}" + (f" ({reason})" if reason else ""))
    hand.update(state=state, since=time.time(), reason=reason, motors_off=False,
                palm_clear_since=None, contact_since=None, auto_armed=False)

def find_fault(act):
    # flags read right after a clear may still be the old ones
    if time.time() - errors_cleared_at < ERROR_CLEAR_GRACE_S:
        return None
    for i, key in enumerate(DOF_KEYS):
        if act["error"][i]:
            return f"{key}: {describe_errors(act['error'][i])}"
    return None

def hand_fully_open(act):
    return all(a >= lim for a, lim in zip(act["angle_act"], OPEN_ANGLES))

def palm_has_object(tactile):
    n, *_ = contact_info(tactile["Palm"])
    return n >= CONTACT_MIN_TAXELS

def any_contact(tactile):
    return any(contact_info(tactile[name])[0] >= CONTACT_MIN_TAXELS for name, *_ in sensor_groups)

# surfaces an object rests on; in an empty fist the fingertips press on each other / the palm
OBJECT_GROUPS = [g[0] for g in sensor_groups if "Pad" in g[0] or g[0] == "Palm"]

def object_contact(tactile):
    return any(contact_info(tactile[name])[0] >= CONTACT_MIN_TAXELS for name in OBJECT_GROUPS)

# ----------------------------
# 👍 THUMB ROTATION GUARD
# ----------------------------
# Runs in the acquisition thread on every frame. When the thumb presses on something, or the
# rotation's force/current leaves its normal range, the rotation target becomes the measured
# angle right away, and write_angles holds back further rotation targets while that lasts.
THUMB_GROUPS = [g[0] for g in sensor_groups if g[4] == "Thumb"]

class RotationGuard:
    def __init__(self):
        self.reason = ""        # why the rotation is stopped; "" = free
        self.stop_angle = None  # angle the rotation was stopped at
        self.force_rest = None  # thumb_rot force reading while the hand is OPENED

    def check(self, act, tactile):
        pressed = sum(int((tactile[g] > ROT_STOP_PRESSURE).sum()) for g in THUMB_GROUPS)
        if pressed >= CONTACT_MIN_TAXELS:
            return f"thumb pressure > {ROT_STOP_PRESSURE} on {pressed} taxels"
        force = act["force_act"][5]
        if self.force_rest is not None and abs(force - self.force_rest) > ROT_STOP_FORCE:
            return f"rotation force {force} (rest {self.force_rest:.0f})"
        if act["current"][5] > ROT_STOP_CURRENT:
            return f"rotation current {act['current'][5]} mA"
        return ""

    def update(self, act, tactile, state):
        if state == "OPENED":   # nothing touches the thumb: learn its resting force reading
            f = act["force_act"][5]
            self.force_rest = f if self.force_rest is None else 0.9 * self.force_rest + 0.1 * f
        reason = "" if state in ("RELEASING", "FAULTY") else self.check(act, tactile)
        if reason and not self.reason:
            self.stop_angle = act["angle_act"][5]
            write_shorts(ACT_SHORT_FIELDS["angle_set"][0] + 2 * 5, [self.stop_angle])
            print(f"Thumb rotation stopped at {self.stop_angle}: {reason}")
        elif self.reason and not reason:
            print("Thumb rotation free again.")
        self.reason = reason

rot_guard = RotationGuard()

def update_hand_state(act, tactile):
    """Called from the GUI loop with the latest fresh frames (None if stale)."""
    if act is not None:
        step_hand_state(act, tactile)
    update_auto_close(tactile)
    update_hand_labels()
    update_object_label(tactile)

def update_object_label(tactile):
    """In HOLD, show 'No object' once no pad or the palm has registered contact for NO_OBJECT_S."""
    now = time.time()
    if hand["state"] != "HOLD" or tactile is None or object_contact(tactile):
        hand["no_contact_since"] = None
    elif hand["no_contact_since"] is None:
        hand["no_contact_since"] = now
    empty = hand["no_contact_since"] is not None and now - hand["no_contact_since"] >= NO_OBJECT_S
    object_label.config(text="⚠ No object in hand" if empty else "")

def warn_hand_state(reason):
    """Software-side problem (e.g. a slow move): shown next to the state, but only the
    hand's error registers make it FAULTY."""
    if hand["reason"] != reason:
        print(f"Hand warning ({hand['state']}): {reason}")
        hand["reason"] = reason

def step_hand_state(act, tactile):
    now = time.time()
    state = hand["state"]
    elapsed = now - hand["since"]

    # the hand counts as stopped once no DOF has moved more than STILL_TOL for STILL_S
    angles = act["angle_act"]
    if hand["last_angles"] is None or max(abs(a - b) for a, b in zip(angles, hand["last_angles"])) > STILL_TOL:
        hand["still_since"] = now
    hand["last_angles"] = angles
    # pending stages and the first frames after a command don't count as "finished"
    stopped = (not pending_moves and elapsed >= STILL_S
               and now - hand["still_since"] >= STILL_S)

    fault = find_fault(act)
    if fault and state != "FAULTY":
        if any(e & (1 << 1) for e in act["error"]):     # ERROR bit 1 = over-temperature
            hold_position(act)                          # stop every DOF from driving
            fault += " – stopped"
        set_hand_state("FAULTY", fault)
        return

    if state == "IDLE":
        if stopped and hand_fully_open(act):
            set_hand_state("OPENED")
    elif state == "RELEASING":
        if not pending_moves and hand_fully_open(act):
            set_hand_state("OPENED")
        elif elapsed > MOVE_TIMEOUT_S:
            stuck = [k for k, a, lim in zip(DOF_KEYS, angles, OPEN_ANGLES) if a < lim]
            warn_hand_state("slow to open: " + ", ".join(stuck))
    elif state in ("CLOSING", "CONTACT"):
        if state == "CLOSING" and tactile is not None and any_contact(tactile):
            set_hand_state("CONTACT")
        elif stopped:
            set_hand_state("HOLD", "" if state == "CONTACT" else "no contact detected")
        elif elapsed > MOVE_TIMEOUT_S:
            warn_hand_state("still moving")
    elif state == "HOLD":
        # stop driving: no motor can keep pushing against the object (or a stall) and overheat;
        # the lead screws keep the grip without power
        if not hand["motors_off"] and elapsed >= HOLD_STOP_S:
            hold_position(act)
            hand["motors_off"] = True
            hand["reason"] = "motors off"
            print("HOLD: motors stopped (targets = measured angles).")

def update_auto_close(tactile):
    """Auto close only fires in OPENED, and only after the palm has been clear,
    so the hand doesn't grab again what it has just let go of."""
    if not auto_hold_var.get() or hand["state"] != "OPENED" or not calibrated or tactile is None:
        return
    now = time.time()
    present = palm_has_object(tactile)
    if not hand["auto_armed"]:
        if present:
            hand["palm_clear_since"] = None
        elif hand["palm_clear_since"] is None:
            hand["palm_clear_since"] = now
        elif now - hand["palm_clear_since"] >= AUTO_HOLD_REARM_S:
            hand["auto_armed"] = True
    elif not present:
        hand["contact_since"] = None
    elif hand["contact_since"] is None:
        hand["contact_since"] = now
    elif now - hand["contact_since"] >= AUTO_HOLD_TRIGGER_S:
        print("Auto close: object detected on palm.")
        close_hand()

def update_hand_labels():
    state = hand["state"]
    text = f"State: {state}" + (f"  ({hand['reason']})" if hand["reason"] else "")
    hand_state_label.config(text=text, fg=HAND_STATE_COLORS[state])

    if not auto_hold_var.get():
        auto_text = "Off"
    elif not calibrated:
        auto_text = "Waiting for sensor data..."
    elif state != "OPENED":
        auto_text = f"Paused – hand is {state} (runs only when OPENED)"
    elif not hand["auto_armed"]:
        auto_text = "Waiting for the palm to be clear..."
    else:
        auto_text = "Armed – place an object on the palm"
    auto_hold_label.config(text=auto_text)
    rot_label.config(text=f"⚠ Thumb rotation stopped: {rot_guard.reason}" if rot_guard.reason else "")

def toggle_auto_hold():
    hand.update(palm_clear_since=None, contact_since=None, auto_armed=False)

def step_plus():
    for s in sliders:
        s.set(min(1000, s.get()+20))
    send_angles()

def step_minus():
    for s in sliders:
        s.set(max(0, s.get()-20))
    send_angles()

# ----------------------------
# 💾 SAVE FUNCTION
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
    with open(SAVE_FILE, "w") as f:
        json.dump(positions, f)

def load_positions():
    if not os.path.exists(SAVE_FILE):
        return

    with open(SAVE_FILE, "r") as f:
        data = json.load(f)

    for name, values in data.items():
        positions[name] = values
        create_button(name, values)

# ----------------------------
# VISUALISATION
# ----------------------------
# The heatmap rectangles are created once and only recoloured afterwards,
# which keeps the GUI thread light so it doesn't slow down acquisition.
cells = {}          # name -> 2D list of canvas item ids
cell_colors = {}    # name -> 2D list of current colours
contact_dots = {}   # name -> canvas item id of the contact marker

def color_for(value):
    v = value / 4000
    if v < 0.2:
        return "#00aa00"
    elif v < 0.5:
        return "#cccc00"
    elif v < 0.8:
        return "#ff8800"
    return "#ff0000"

def build_canvas():
    canvas.create_image(350, 450, image=hand_img)
    for name, addr, h, w, key, rel in sensor_groups:
        x0 = offsets[key][0] + rel[0]
        y0 = offsets[key][1] + rel[1]
        cells[name] = [[
            canvas.create_rectangle(
                x0+j*pixel, y0+i*pixel,
                x0+(j+1)*pixel, y0+(i+1)*pixel,
                fill="#00aa00", width=0, stipple="gray25"
            ) for j in range(w)] for i in range(h)]
        cell_colors[name] = [["#00aa00"] * w for _ in range(h)]
        contact_dots[name] = canvas.create_oval(0, 0, 0, 0, fill="blue", state="hidden")

def draw(tactile):
    for name, addr, h, w, key, rel in sensor_groups:
        matrix = tactile[name]

        # Heatmap
        for i in range(h):
            for j in range(w):
                c = color_for(matrix[i, j])
                if c != cell_colors[name][i][j]:
                    canvas.itemconfig(cells[name][i][j], fill=c)
                    cell_colors[name][i][j] = c

        # ✅ ROBUST CONTACT
        n, cy, cx, force = contact_info(matrix)
        dot = contact_dots[name]
        if cy is None:
            canvas.itemconfig(dot, state="hidden")
            continue

        x0 = offsets[key][0] + rel[0] + int(cx)*pixel
        y0 = offsets[key][1] + rel[1] + int(cy)*pixel
        size = int(3 + force/200)  # smooth scaling
        canvas.coords(dot, x0, y0, x0+size, y0+size)
        canvas.itemconfig(dot, state="normal")

def describe_errors(code):
    return ", ".join(n for bit, n in ERROR_BITS.items() if code & (1 << bit)) or f"0x{code:02x}"

def update_status_label(act):
    if act is None:
        status_label.config(text="No actuator data", fg="black")
        return
    # target = angle_set as read back from the hand, i.e. what it is actually driving to;
    # force = force_act / force_set, the finger stops once the first reaches the second
    lines = [f"{'DOF':<10} {'real/target':>11} {'force/limit':>11} {'status':<13} {'current':>6} {'temp':>5}"]
    for i, key in enumerate(DOF_KEYS):
        status = STATUS_NAMES.get(act["status"][i], str(act["status"][i]))
        angles = f"{act['angle_act'][i]}/{act['angle_set'][i]}"
        forces = f"{act['force_act'][i]}/{act['force_set'][i]}"
        line = (f"{key:<10} {angles:>11} {forces:>11} {status:<13} "
                f"{act['current'][i]:>4}mA {act['temp'][i]:>3}°C")
        if act["error"][i]:
            line += f"  ERR: {describe_errors(act['error'][i])}"
        if act["temp"][i] > TEMP_VALID_MAX:
            line += "  invalid temp"
        lines.append(line)
    valid_temps = [t for t in act["temp"] if t <= TEMP_VALID_MAX]
    problem = any(act["error"]) or max(valid_temps, default=0) >= TEMP_WARN_C
    status_label.config(text="\n".join(lines), fg="red" if problem else "black")

# ----------------------------
sliders_initialised = False

def gui_loop():
    global sliders_initialised
    with latest_lock:
        tactile = latest["tactile"]
        act = latest["actuators"]
        t = latest["t"]

    if tactile is not None:
        draw(tactile)

    # don't act on a stale frame if acquisition has stalled
    fresh = time.time() - t < 0.5
    update_hand_state(act if fresh else None, tactile if fresh else None)

    # take the current target angles from the hand once at startup
    if not sliders_initialised and act is not None:
        for i in range(6):
            sliders[i].set(act["angle_set"][i])
        sliders_initialised = True

    update_status_label(act if fresh else None)
    update_log_label()
    if not closing:
        root.after(GUI_RATE_MS, gui_loop)

# ----------------------------
# UI
# ----------------------------
sliders=[]
positions={}

names=["Little finger","Ring finger","Middle finger","Index finger","Thumb bend","Thumb rotate"]

for i in range(6):
    f=tk.Frame(left_frame)
    f.pack()

    tk.Label(f,text=names[i],width=15).pack(side="left")

    s=tk.Scale(f,from_=0,to=1000,orient="horizontal",length=250)
    s.pack(side="left")
    s.bind("<ButtonRelease-1>",lambda e: send_angles())

    sliders.append(s)

# Buttons
bf=tk.Frame(left_frame)
bf.pack(pady=10)

tk.Button(bf,text="Open",command=open_hand).pack(side="left")
tk.Button(bf,text="Close",command=close_hand).pack(side="left")
tk.Button(bf,text="Step +",command=step_plus).pack(side="left")
tk.Button(bf,text="Step -",command=step_minus).pack(side="left")

# ----------------------------
# 🩺 Hand status UI
# ----------------------------
status_frame = tk.LabelFrame(left_frame, text="Hand status")
status_frame.pack(fill="x", pady=10)

ff = tk.Frame(status_frame)
ff.pack(fill="x")
tk.Label(ff, text="Grip force", width=15).pack(side="left")
force_slider = tk.Scale(ff, from_=0, to=3000, orient="horizontal", length=250)
force_slider.set(FORCE_LIMIT)
force_slider.pack(side="left")
force_slider.bind("<ButtonRelease-1>", lambda e: set_force_limit())

hand_state_label = tk.Label(status_frame, text="State: IDLE", font=("TkDefaultFont", 12, "bold"),
                            justify="left")
hand_state_label.pack(anchor="w")

object_label = tk.Label(status_frame, text="", fg="#cc7700", font=("TkDefaultFont", 11, "bold"))
object_label.pack(anchor="w")

rot_label = tk.Label(status_frame, text="", fg="#cc00cc", justify="left", wraplength=480)
rot_label.pack(anchor="w")

tk.Button(status_frame, text="Clear errors", command=clear_faults).pack(fill="x")

status_label = tk.Label(status_frame, text="No actuator data", justify="left",
                        font="TkFixedFont")
status_label.pack(anchor="w")

# ----------------------------
# 🤚 Auto hold UI
# ----------------------------
auto_hold_frame = tk.LabelFrame(left_frame, text="Auto hold")
auto_hold_frame.pack(fill="x", pady=10)

auto_hold_var = tk.BooleanVar(value=False)
tk.Checkbutton(auto_hold_frame, text="Close hand when an object touches the palm",
               variable=auto_hold_var, command=toggle_auto_hold).pack(anchor="w")

auto_hold_label = tk.Label(auto_hold_frame, text="Off", justify="left")
auto_hold_label.pack(anchor="w")

# ----------------------------
# 📈 Logging UI
# ----------------------------
log_frame = tk.LabelFrame(left_frame, text="Data logging")
log_frame.pack(fill="x", pady=10)

log_button = tk.Button(log_frame, text="● Start logging", command=toggle_logging)
log_button.pack(fill="x")
log_button_bg = log_button.cget("bg")

log_label = tk.Label(log_frame, text="Logging off", justify="left")
log_label.pack(fill="x")

# ----------------------------
# 💾 Save UI
# ----------------------------
entry = tk.Entry(left_frame)
entry.pack(fill="x", pady=5)

tk.Button(left_frame, text="Save position", command=save_position).pack(fill="x")

saved_frame = tk.Frame(left_frame)
saved_frame.pack(fill="both", expand=True)

load_positions()

# ----------------------------
# Canvas
# ----------------------------
canvas=tk.Canvas(right_frame,width=700,height=900)
canvas.pack()
build_canvas()

closing = False

def on_close():
    global closing
    if closing:
        return
    closing = True
    stop_event.set()
    acq_thread.join(timeout=2)
    logger.stop()
    client.close()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)

# Ctrl+C in the terminal closes the window cleanly (log files are saved)
signal.signal(signal.SIGINT, lambda *args: root.after(0, on_close))

set_force_limit(FORCE_LIMIT)
write_shorts(ACT_SHORT_FIELDS["speed_set"][0], [SPEED_FINGERS] * 4 + [SPEED_THUMB] * 2)
acq_thread.start()
gui_loop()
root.mainloop()
