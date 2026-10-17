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

FORCE_LIMIT = 2500         # force_set of the fingers (0–3000) for the grip (Grip force slider)
WRITE_SAFE_FORCE = 1500    # no writes but Open / Clear errors while a finger presses this hard: in
                           # logs 20261017_* 0 of 155 writes below 1500 g froze the motor controller,
                           # 3 of 10 at 1500–2000 g and 11 of 213 at >= 2000 g
CLOSE_FORCE_LIMIT = 1000   # force_set of the fingers during Close until its last stage (full grip)
THUMB_ROT_MID = 300       # Close: thumb rotation goes out to this before the middle finger closes ...
THUMB_ROT_SLOW_SPEED = 100 # ... and then to 1000 at this speed (0–1000) before the index closes
                           # (100: 300 -> 1000 takes ~4 s; at 200 it pushed the object 300 further)
THUMB_DONE_TIMEOUT_S = 8.0 # the index waits for that rotation to stop (target or object), at most this
STAGE_DELAY_MS = 300       # pause between the Close stages
THUMB_RETURN_WAIT_MS = 1000  # Close: wait after the middle closes before the thumb rotates to 1000
STAGE_WAIT_S = 3.0         # a stage waits at most this long for every finger to press < WRITE_SAFE_FORCE
FORCE_LIMIT_THUMB_BEND = 1000  # force_set of thumb bending
FORCE_LIMIT_THUMB_ROT = 100    # force_set of thumb rotation: low, so it stops softly on the object
                               # (its force reading at rest has been ~50, but also ~440 – then it can't move)
OPEN_FORCE_LIMIT = 3000    # force_set while opening: releasing squeezes nothing (hand default)
SPEED_FINGERS = 500        # speed_set of the fingers (0–1000; 1000 = full stroke in ~0.6 s);
                           # slower = less force overshoot when a finger hits the object
SPEED_THUMB = 1000          # slower thumb: gentler contact, lower current, time for the guard to react

# Thumb rotation guard: the hand's force limit never stops a rotation that pushes the object
# sideways, so the software stops it when one of these trips. (Its force reading isn't used: it
# reads -250…-340 while the thumb rotates freely and keeps 200–600 off rest after a move with no
# current, so it stopped free moves at 77, 206 and 263 in logs/20261017_190436.)
ROT_STOP_PRESSURE = 500    # thumb taxel value above baseline ...
ROT_STOP_TAXELS = 5        # ... on at least this many thumb taxels
ROT_MOVE_CURRENT = 150     # |thumb_rot current| while it still rotates that counts as pushing the
ROT_MOVE_PUSH_S = 0.2      # object along, for this long. Free rotation at speed 200 read 40–140 mA;
                           # pushing in logs/20261017_192143 read 250–730 mA from angle ~520 on, while
                           # its sensors showed nothing until 891. Lower = stops earlier (at 0.1 s it
                           # also tripped on normal fast moves in the logs)
ROT_MOVE_SKIP_S = 0.15     # not counted this long after a rotation command (start spike) ...
ROT_MOVE_NEAR = 60         # ... nor this close to the target (braking spike)
ROT_STOP_CURRENT = 300     # |thumb_rot current| in mA, counted only while the thumb isn't moving:
                           # it reads 300–800 mA for 1–4 frames when it brakes at its target or starts
                           # a move (logs 20261017_*); pushing read -300…-440 mA pulses while still
ROT_STILL_S = 0.15         # "not moving" = the rotation angle changed by at most ROT_STILL_TOL
ROT_STILL_TOL = 10         # over the last ROT_STILL_S
ROT_PUSH_S = 0.2           # current above ROT_STOP_CURRENT this long after the first such reading,
                           # with the thumb still the whole time, counts as pushing
ROT_STALL_S = 0.4          # "can't rotate further": moved at most ROT_STILL_TOL in this long, at least
ROT_STALL_TOL = 20         # this long after its last command, and still more than this short of the
                           # target (at speed 200 it pauses up to 0.15 s between steps while moving)
# Slip detection (HOLD only, shown in the GUI, not logged): each surface the object touches is
# compared with how it touched once the grip has settled
SLIP_SETTLE_S = 0.5        # settled = no finger / thumb bend moved more than SLIP_SETTLE_TOL in
SLIP_SETTLE_TOL = 5        # this long (the index keeps creeping into the object for a median 1.8 s,
SLIP_SETTLE_MAX_S = 3.0    # up to 6 s, after it closed); at the latest this long into HOLD
SLIP_THRESHOLD = 100       # taxel value above baseline that counts (the pads drift up to ~90)
SLIP_MIN_TAXELS = 5        # a contact centre needs this many such taxels (3x3 tips: never)
SLIP_MIN_PRESSURE = 1000   # a surface is watched for a pressure drop once it carries this much (sum
                           # of its taxels above SLIP_THRESHOLD); fingertips at 5 taxels / ~800
                           # flickered between 4 and 5 taxels without losing pressure
SLIP_WINDOW_S = 1.0        # contact is compared with this long ago: a slip is sudden, the grip
                           # relaxing (fingertip pressure -60% over 4–8 s in the logs) isn't
SLIP_SHIFT = 2.0           # contact centre moved this many taxels within SLIP_WINDOW_S -> slip
SLIP_PRESSURE_DROP = 0.4   # surface pressure fell below this fraction within SLIP_WINDOW_S -> slip
SLIP_ANGLE = 30            # a finger / the thumb bend pushed this far OPEN (0–1000) -> slip. Closing
                           # further isn't: the index keeps creeping 25–250 into the object after
                           # the grip looks finished (logs 20261017_20*); the thumb rotation is
                           # left out, the rotation guard moves it on purpose
TEMP_WARN_C = 60           # highlight the status readout above this motor temperature
TEMP_VALID_MAX = 100       # manual: TEMP range is 0–100 °C, above is a corrupted reading (display only)

# Hand state machine
# a DOF at or above its angle counts as fully open (order of DOF_KEYS);
# thumb_rot is lower because it currently stops at ~817 instead of 1000
OPEN_ANGLES = [950, 950, 950, 950, 950, 750]
STILL_TOL = 3              # angle change per GUI tick below which a DOF counts as not moving
STILL_S = 0.4              # the hand must be still this long for a move to count as finished
NO_OBJECT_S = 0.5          # no pad/palm contact this long in HOLD -> "No object" shown
FREEZE_S = 0.3             # motor feedback unchanged this long while frames keep arriving = the hand's
                           # motor controller is frozen (live feedback stayed unchanged <= 0.26 s in
                           # the logs, bar two runs of 0.41/0.44 s)
MOVE_TIMEOUT_S = 3.0      # opening/closing that takes longer than this -> warning (not FAULTY)
ERROR_CLEAR_GRACE_S = 0.5  # ignore error flags this long after clearing them
STALL_RETRIES = 3          # a stall-only fault is cleared and the move sent again at most this often
STALL_RETRY_WINDOW_S = 10.0  # ... within this long; one more and the hand goes FAULTY as before
                           # (logs/20261017_201853: the index flagged a stall while idle and open,
                           # stayed in status 7, and moved normally on the next command)

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
AUTO_HOLD_REARM_S = 1.0      # palm and pads must stay clear this long before auto hold can trigger again
# surfaces whose contact (>= CONTACT_MIN_TAXELS taxels) starts auto close; the finger pads read <= 2
# taxels while the hand stood open in logs/20261017_*, the fingertips ("Nail") up to 5, so not those
AUTO_CLOSE_GROUPS = ["Palm", "Little Pad", "Ring Pad", "Middle Pad", "Index Pad"]
AUTO_CLOSE_THRESHOLD = 100   # taxel value above baseline that counts for auto close / auto record.
                             # At CONTACT_THRESHOLD (30) the pads' baseline drift (58–84 on 4–5 index
                             # pad taxels, 20:23–20:24) closed the hand on nothing 4 times; at 100 none
                             # of those trigger and all 36 real objects still do (logs 20261017_20*)
AUTO_RECORD_HOLD_S = 2.0     # Auto record: logging stops this long after the grip is finished
                             # (HOLD, i.e. the index closed and nothing moves), then the hand opens
AUTO_RECORD_FROZEN_S = 2.0   # ... or after the motor data has been frozen this long, and opens anyway
                             # (a frozen hand carries the Open out once it recovers, or once the
                             # object is taken out)

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
latest = {"tactile": None, "actuators": None, "t": 0.0, "act_changed_t": 0.0}
ACT_FEEDBACK = ("pos_act", "angle_act", "force_act", "current", "temp")
stats = {"rate_hz": 0.0, "errors": 0}

def acquisition_loop():
    calibrate_offsets()
    if LOG_AUTOSTART:
        root.after(0, toggle_logging)

    period = 1.0 / SAMPLE_RATE_HZ
    next_t = time.perf_counter()
    last_t = None
    last_fb, act_changed_t = None, time.time()

    while not stop_event.is_set():
        t = time.time()
        raw = read_tactile_raw()
        act = read_actuators()

        if raw is None or act is None:
            stats["errors"] += 1
        else:
            tactile = correct_tactile(raw)
            # live motor feedback always changes a little within FREEZE_S (force/current noise)
            fb = tuple(tuple(act[k]) for k in ACT_FEEDBACK)
            if fb != last_fb:
                last_fb, act_changed_t = fb, t
            with latest_lock:
                latest["tactile"] = tactile
                latest["actuators"] = act
                latest["t"] = t
                latest["act_changed_t"] = act_changed_t
            logger.log(t, act, tactile)
            if t - act_changed_t <= FREEZE_S:      # no decisions on frozen motor data
                rot_guard.update(act, tactile, hand["state"])
            slip.update(act, tactile, hand["state"], hand["since"], t - act_changed_t <= FREEZE_S)

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

def start_logging():
    if not calibrated:
        print("Note: sensors not calibrated yet – offsets are 0.")
    logger.start()
    log_button.config(text="■ Stop logging", bg="#ff6666")

def stop_logging():
    logger.stop()
    log_button.config(text="● Start logging", bg=log_button_bg)
    hand.update(auto_rec=False, auto_rec_hold_t=None)

def toggle_logging():
    if logger.active:
        stop_logging()
    else:
        start_logging()

def update_log_label():
    rate = f"Acquisition: {stats['rate_hz']:.1f} Hz (target {SAMPLE_RATE_HZ})"
    if stats["errors"]:
        rate += f", {stats['errors']} read errors"
    if hand["auto_rec"]:
        t = hand["auto_rec_hold_t"]
        rate += ("\nAuto record: recording until the grip is finished" if t is None else
                 f"\nAuto record: stops and opens in {max(0.0, AUTO_RECORD_HOLD_S - (time.time() - t)):.1f} s")
    elif auto_record_var.get():
        rate += "\nAuto record: waiting for an object"
    if logger.active:
        log_label.config(text=f"{rate}\n{os.path.relpath(logger.session_dir, BASE_DIR)}\n"
                              f"{logger.frame} frames, {time.time() - logger.t0:.1f} s")
    else:
        log_label.config(text=f"{rate}\nLogging off")

# ----------------------------
# CONTROL
# ----------------------------
write_skip_printed_at = 0.0
last_write_t = 0.0          # when the last write went out (stages wait for the hand to respond)

def finger_load(act):
    """Highest force reading of the four fingers (g)."""
    return max(act["force_act"][:4])

def write_shorts(addr, values, force=False):
    """Writes registers; returns False (and prints why) if the write failed or was skipped.
    While a finger presses with WRITE_SAFE_FORCE or more, only force=True writes (Open, Clear
    errors) go out: a write while the fingers grip can freeze the hand's motor controller, and a
    frozen hand doesn't let the clamped fingers go. In logs/20261017_* (until 20:32) that
    happened after 1 of 428 writes with every finger below 500 g, 3 of 87 at 500–1500 g, 3 of 40
    at 1500–2000 g and 11 of 269 at 2000 g or more."""
    global write_skip_printed_at, last_write_t
    if not force:
        with latest_lock:
            act = latest["actuators"]
        if act is not None and finger_load(act) >= WRITE_SAFE_FORCE:
            if time.time() - write_skip_printed_at > 1.0:
                write_skip_printed_at = time.time()
                print(f"Write to {addr} skipped: a finger presses {finger_load(act)} g "
                      f"(>= {WRITE_SAFE_FORCE}) – only Open goes out now.")
            return False
    try:
        with modbus_lock:
            res = client.write_registers(address=addr, values=values, device_id=1)
    except Exception as e:      # pymodbus raises on connection problems
        print(f"Write to {addr} failed: {e}")
        return False
    if res.isError():
        print(f"Write to {addr} failed: {res}")
        return False
    last_write_t = time.time()
    return True

errors_cleared_at = 0.0

def clear_errors():
    global errors_cleared_at
    if write_shorts(CLEAR_ERROR_ADDR, [1], force=True):
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

def target_angles():
    targets = [s.get() for s in sliders]
    if rot_guard.reason and hand["state"] != "RELEASING":
        # thumb is pressing: keep the rotation where the guard stopped it (Open is exempt)
        targets[5] = rot_guard.stop_angle
        sliders[5].set(targets[5])
    return targets

def write_angles(force=False):
    # latched errors are never cleared here – only the Clear errors button does that
    write_shorts(ACT_SHORT_FIELDS["angle_set"][0], target_angles(), force)

def write_pose(angles, limits, force=False):
    """Targets and force limits in ONE write: angle_set (1486) and force_set (1498) are adjacent,
    so 12 registers from 1486 set both. Each extra write while the fingers grip is a chance to
    freeze the hand (see write_shorts)."""
    for s, a in zip(sliders, angles):
        s.set(a)
    write_shorts(ACT_SHORT_FIELDS["angle_set"][0], list(angles) + list(limits), force)

def set_speeds(fingers=SPEED_FINGERS, thumb_rot=SPEED_THUMB, force=False):
    write_shorts(ACT_SHORT_FIELDS["speed_set"][0], [fingers] * 4 + [SPEED_THUMB, thumb_rot], force)

def hold_position(act):
    """Stop driving: every target becomes the measured angle, so the motors stop pushing
    (the lead screws keep the fingers where they are)."""
    cancel_pending_moves()
    for s, a in zip(sliders, act["angle_act"]):
        s.set(a)
    write_shorts(ACT_SHORT_FIELDS["angle_set"][0], list(act["angle_act"]))

def motion_blocked(gripping_ok=False):
    """True (and the command is dropped) while the hand is FAULTY, or – unless gripping_ok –
    while a finger presses with WRITE_SAFE_FORCE or more (only Open may be sent then)."""
    with latest_lock:
        act = latest["actuators"]
    if hand["state"] == "FAULTY":
        print(f"Command ignored: hand is FAULTY ({hand['reason']}) – press Clear errors first.")
    elif not gripping_ok and act is not None and finger_load(act) >= WRITE_SAFE_FORCE:
        print(f"Command ignored: a finger presses {finger_load(act)} g – press Open first "
              "(writes at this force can freeze the hand).")
    else:
        return False
    if act is not None:         # put the sliders back to what the hand is really set to
        for s, a in zip(sliders, act["angle_set"]):
            s.set(a)
    return True

# Close moves in stages; any new command cancels the stages still pending
pending_moves = []

def cancel_pending_moves():
    for after_id in pending_moves:
        root.after_cancel(after_id)
    pending_moves.clear()

def run_stages(stages):
    """stages: list of (action, wait_ms[, ready[, send_loaded]]): action is {dof_index: angle} or
    a function to call; the next stage follows wait_ms later. With ready (a function), the stage
    first waits up to THUMB_DONE_TIMEOUT_S for ready() to be true. A stage only goes out while
    every finger presses less than WRITE_SAFE_FORCE: it waits up to STAGE_WAIT_S for that, then
    the sequence stops there. A send_loaded stage doesn't wait: it goes out right away, with a
    warning if a finger is loaded."""
    cancel_pending_moves()
    def apply(k, due=None, load_due=None):
        pending_moves.clear()       # this stage is running now
        now = time.time()
        with latest_lock:
            frozen = now - latest["t"] > 0.5 or latest["t"] - latest["act_changed_t"] > FREEZE_S
            answered = latest["act_changed_t"] > last_write_t
        if not frozen and not answered:
            # the motor readings haven't changed since the last write yet (normally < 0.1 s):
            # a write into the first FREEZE_S of a freeze would only queue up (logs/203103)
            pending_moves.append(root.after(30, apply, k, due, load_due))
            return
        if frozen:
            # nothing to decide on (and a write now would only queue up): wait for live motor
            # data, then this stage's waits start over
            warn_hand_state(f"close paused before stage {k + 1}/{len(stages)}: motor data frozen")
            pending_moves.append(root.after(100, apply, k))
            return
        due = now if due is None else due
        action, wait_ms, ready, send_loaded = (tuple(stages[k]) + (None, False))[:4]
        if ready is not None and not ready() and now - due < THUMB_DONE_TIMEOUT_S:
            pending_moves.append(root.after(100, apply, k, due, load_due))   # keep both waits
            return
        with latest_lock:
            act = latest["actuators"]
        if act is not None and finger_load(act) >= WRITE_SAFE_FORCE:
            if not send_loaded:
                load_due = now if load_due is None else load_due
                if now - load_due < STAGE_WAIT_S:
                    pending_moves.append(root.after(100, apply, k, due, load_due))
                    return
                warn_hand_state(f"close stopped before stage {k + 1}/{len(stages)}: "
                                f"a finger presses {finger_load(act)} g")
                return
            print(f"WARNING: Close stage {k + 1}/{len(stages)} sent while a finger presses "
                  f"{finger_load(act)} g – this write can freeze the hand (then Open only "
                  "works once the object is taken out).")
        if callable(action):
            action()
        else:
            if 5 in action:
                rot_guard.release()
            for i, v in action.items():
                sliders[i].set(v)
            write_angles()
        if k + 1 < len(stages):
            pending_moves.append(root.after(wait_ms, apply, k + 1))
    apply(0)

def send_angles():
    """Manual command (slider, Step, saved position): the hand leaves its open/closed state."""
    if motion_blocked():
        return
    cancel_pending_moves()
    rot_guard.release()
    set_move_limits([s.get() for s in sliders])
    write_angles()
    set_hand_state("IDLE", "manual pose")

def open_hand():
    if motion_blocked(gripping_ok=True):
        return
    cancel_pending_moves()
    # RELEASING first, so the rotation guard leaves the thumb alone; Open always goes out, even
    # at full grip (force=True), as ONE write. OPEN_FORCE_LIMIT: the grip force limit would stop
    # a DOF whose force reading rises while opening
    set_hand_state("RELEASING")
    write_pose([1000] * 6, [OPEN_FORCE_LIMIT] * 6, force=True)

def close_hand():
    """Thumb in (rotation and bend 0), little + ring, thumb rotation out to THUMB_ROT_MID, middle,
    THUMB_RETURN_WAIT_MS later (and once the thumb has stopped) the thumb rotation slowly to
    1000, and once that rotation has stopped (at its target, on the object, or by the guard) the
    index. The fingers close with only CLOSE_FORCE_LIMIT until the last stage, which closes the
    index and sets the full grip force in one write: every stage is a write, and writes while a
    finger grips hard can freeze the hand. The last two stages go out even then (with a
    warning). The speeds go back to normal once the hand is open again."""
    if motion_blocked():
        return
    set_speeds()
    set_force_limit(CLOSE_FORCE_LIMIT)
    def middle():
        sliders[2].set(0)
        write_angles()
        # the slow thumb speed goes out now, while the fingers are still moving (180–330 g in the
        # logs), not with the thumb stage below when they grip: one write less under load
        set_speeds(thumb_rot=THUMB_ROT_SLOW_SPEED)
    def thumb_slow():
        # run_stages has already waited for the load; it goes out even at full load (force=True)
        with latest_lock:
            act = latest["actuators"]
        if act is None or act["speed_set"][5] != THUMB_ROT_SLOW_SPEED:   # middle() couldn't set it
            set_speeds(thumb_rot=THUMB_ROT_SLOW_SPEED, force=True)
        rot_guard.release()
        sliders[5].set(1000)
        write_angles(force=True)
    def thumb_done():
        return rot_guard.settled()
    def index_full_grip():
        # run_stages has already waited for the load; it goes out even at full load (force=True)
        sliders[3].set(0)
        write_shorts(ACT_SHORT_FIELDS["angle_set"][0], target_angles() + grasp_limits(), force=True)
    set_hand_state("CLOSING")
    run_stages([
        ({5: 0, 4: 0}, STAGE_DELAY_MS),
        ({0: 0, 1: 0}, STAGE_DELAY_MS),
        ({5: THUMB_ROT_MID}, STAGE_DELAY_MS),
        (middle, THUMB_RETURN_WAIT_MS),
        (thumb_slow, STAGE_DELAY_MS, thumb_done, True),
        (index_full_grip, 0, thumb_done, True),
    ])

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
    "clear_since": None, "contact_since": None,         # auto close timers
    "auto_armed": False,
    "auto_rec": False, "auto_rec_hold_t": None,         # Auto record: session it started, HOLD since
    "frozen": False,                                    # motor feedback currently frozen
    "stall_retries": [],                                # times of automatic stall clears
    "no_contact_since": None,                           # "No object" display in HOLD
}

def set_hand_state(state, reason=""):
    if state != hand["state"]:
        print(f"Hand state: {hand['state']} -> {state}" + (f" ({reason})" if reason else ""))
        if state == "OPENED":
            set_speeds()        # Close slows the thumb rotation; reset while the hand is empty
    hand.update(state=state, since=time.time(), reason=reason,
                clear_since=None, contact_since=None, auto_armed=False)

def find_fault(act):
    # flags read right after a clear may still be the old ones
    if time.time() - errors_cleared_at < ERROR_CLEAR_GRACE_S:
        return None
    for i, key in enumerate(DOF_KEYS):
        if act["error"][i]:
            return f"{key}: {describe_errors(act['error'][i])}"
    return None

def retry_stall(act, now):
    """Only stall flags (ERROR bit 0) set: clear them and send the current targets again, up to
    STALL_RETRIES times within STALL_RETRY_WINDOW_S. True if it did."""
    if not all(e & ~1 == 0 for e in act["error"]):     # any other fault bit: FAULTY as before
        return False
    recent = [t for t in hand["stall_retries"] if now - t < STALL_RETRY_WINDOW_S]
    if len(recent) >= STALL_RETRIES:
        print(f"Stall again after {STALL_RETRIES} automatic clears within "
              f"{STALL_RETRY_WINDOW_S:.0f} s – giving up.")
        return False
    hand["stall_retries"] = recent + [now]
    stalled = ", ".join(k for k, e in zip(DOF_KEYS, act["error"]) if e)
    print(f"Stall on {stalled}: clearing the fault and trying again "
          f"({len(recent) + 1}/{STALL_RETRIES}).")
    clear_errors()
    write_angles()              # the move again (skipped while a finger grips too hard)
    warn_hand_state(f"stall on {stalled} cleared, retried")
    return True

def hand_fully_open(act):
    return all(a >= lim for a, lim in zip(act["angle_act"], OPEN_ANGLES))

def auto_close_contact(tactile):
    """Name of the first AUTO_CLOSE_GROUPS surface with CONTACT_MIN_TAXELS taxels above
    AUTO_CLOSE_THRESHOLD, or None."""
    return next((name for name in AUTO_CLOSE_GROUPS
                 if int((tactile[name] > AUTO_CLOSE_THRESHOLD).sum()) >= CONTACT_MIN_TAXELS), None)

def any_contact(tactile):
    return any(contact_info(tactile[name])[0] >= CONTACT_MIN_TAXELS for name, *_ in sensor_groups)

# surfaces an object rests on; in an empty fist the fingertips press on each other / the palm
OBJECT_GROUPS = [g[0] for g in sensor_groups if "Pad" in g[0] or g[0] == "Palm"]

def object_contact(tactile):
    return any(contact_info(tactile[name])[0] >= CONTACT_MIN_TAXELS for name in OBJECT_GROUPS)

# ----------------------------
# 👍 THUMB ROTATION GUARD
# ----------------------------
# Runs in the acquisition thread on every frame. When the thumb presses on something or can't
# rotate further, the rotation target becomes the measured angle right away and stays there (also
# through write_angles) until the next rotation command (release()) or Open: it stops, it doesn't
# keep forcing.
THUMB_GROUPS = [g[0] for g in sensor_groups if g[4] == "Thumb"]

class RotationGuard:
    def __init__(self):
        self.reason = ""        # why the rotation is stopped; "" = free
        self.stop_angle = None  # angle the rotation was stopped at
        self.angles = []        # (time, rotation angle) of the last ROT_STALL_S
        self.push_since = None  # first high current while the thumb is still; None = moving
        self.target = None      # rotation target read back from the hand ...
        self.target_t = 0.0     # ... and when it last changed
        self.release_angle = None  # angle at release(); pressure only counts once it moved away
        self.move_push_since = None  # high current while rotating since; None = not pushing

    def moving(self, window=ROT_STILL_S):
        now = time.time()
        angles = [a for t, a in self.angles if now - t <= window]
        return bool(angles) and max(angles) - min(angles) > ROT_STILL_TOL

    def settled(self):
        """The thumb rotation has stopped: at its target, on the object, or stopped here.
        False on old samples: update() doesn't run while the motor feedback is frozen. (Frames
        can be 0.24 s apart when the PC is busy, so "old" is FREEZE_S, not one frame.)"""
        return (bool(self.angles) and time.time() - self.angles[-1][0] <= FREEZE_S
                and not self.moving(ROT_STALL_S))

    def release(self):
        """A new rotation command (GUI thread): the thumb may move again."""
        if self.reason:
            print("Thumb rotation free again (new rotation command).")
        self.reason = ""
        self.push_since = None
        self.move_push_since = None
        self.release_angle = self.angles[-1][1] if self.angles else None

    def check(self, act, tactile):
        now, angle = time.time(), act["angle_act"][5]
        if self.release_angle is not None and abs(angle - self.release_angle) > ROT_STALL_TOL:
            self.release_angle = None
        pressed = sum(int((tactile[g] > ROT_STOP_PRESSURE).sum()) for g in THUMB_GROUPS)
        if pressed >= ROT_STOP_TAXELS and self.release_angle is None:
            return f"thumb pressure > {ROT_STOP_PRESSURE} on {pressed} taxels"
        current = act["current"][5]
        if self.move_push_since is not None and now - self.move_push_since >= ROT_MOVE_PUSH_S:
            return f"pushing while rotating ({current} mA at {angle})"
        # either direction; braking at a target or starting a move gives a short spike, no push
        if (abs(current) > ROT_STOP_CURRENT and self.push_since is not None
                and now - self.push_since >= ROT_PUSH_S):
            return f"rotation current {current} mA (thumb not moving)"
        if (now - self.target_t >= ROT_STALL_S and self.settled()
                and now - self.angles[0][0] >= ROT_STALL_S - 0.05
                and abs(self.target - angle) > ROT_STALL_TOL):
            return f"can't rotate further ({angle}, target {self.target})"
        return ""

    def update(self, act, tactile, state):
        now = time.time()
        self.angles = [(t, a) for t, a in self.angles if now - t <= ROT_STALL_S]
        self.angles.append((now, act["angle_act"][5]))
        if act["angle_set"][5] != self.target:
            self.target, self.target_t = act["angle_set"][5], now
        pushing = (abs(act["current"][5]) > ROT_MOVE_CURRENT and now - self.target_t >= ROT_MOVE_SKIP_S
                   and abs(self.target - act["angle_act"][5]) > ROT_MOVE_NEAR)
        if not pushing:
            self.move_push_since = None
        elif self.move_push_since is None:
            self.move_push_since = now
        if self.moving():
            self.push_since = None
        elif abs(act["current"][5]) > ROT_STOP_CURRENT and self.push_since is None:
            self.push_since = now
        if state in ("RELEASING", "FAULTY"):
            if self.reason:
                print("Thumb rotation free again.")
            self.reason = ""
            return
        if self.reason:         # stopped: stays stopped until release() or Open
            return
        reason = self.check(act, tactile)
        if reason:
            self.stop_angle = act["angle_act"][5]
            if act["angle_set"][5] != self.stop_angle:     # already there: no extra write
                # sent even while the fingers grip hard: not stopping would keep forcing the thumb
                if finger_load(act) >= WRITE_SAFE_FORCE:
                    print(f"WARNING: thumb stop sent while a finger presses {finger_load(act)} g "
                          "– this write can freeze the hand.")
                write_shorts(ACT_SHORT_FIELDS["angle_set"][0] + 2 * 5, [self.stop_angle], force=True)
            print(f"Thumb rotation stopped at {self.stop_angle}: {reason} – holds until the "
                  "next rotation command.")
            self.reason = reason

rot_guard = RotationGuard()

# ----------------------------
# 🫳 SLIP DETECTION
# ----------------------------
# Runs in the acquisition thread on every frame (the tactile data stays live even when the motor
# feedback freezes). Only reports – it never writes to the hand: a write while the fingers grip
# can freeze the motor controller.
def palm_physical(m):
    """The palm's data is column-major and bottom-to-top (data point k = row 7 - k%8, col k//8),
    but correct_tactile reshapes it row-major: put it back in physical order for centroids."""
    flat = np.asarray(m).ravel()
    out = np.zeros((8, 14))
    k = np.arange(flat.size)
    out[7 - k % 8, k // 8] = flat
    return out

class SlipDetector:
    def __init__(self):
        self.reset()

    def reset(self):
        self.watching = False   # the grip has settled, contact is being compared
        self.history = []       # (time, {surface: (centroid row, col, pressure)}) of SLIP_WINDOW_S
        self.ref_angles = None  # finger angles once the grip settled
        self.angles = []        # (time, angles) of the last SLIP_SETTLE_S, while the motors are live
        self.reasons = []       # why it slipped (latched until the hand leaves HOLD)
        self.status = ""        # text for the GUI

    @staticmethod
    def surface(m):
        """(centroid, pressure) of the taxels above SLIP_THRESHOLD; the centroid (row, col) is
        None with fewer than SLIP_MIN_TAXELS of them."""
        w = np.where(m > SLIP_THRESHOLD, m, 0).astype(float)
        total = w.sum()
        if int((w > 0).sum()) < SLIP_MIN_TAXELS:
            return None, total
        rows, cols = np.indices(w.shape)
        return ((rows * w).sum() / total, (cols * w).sum() / total), total

    def surfaces(self, tactile):
        out = {}
        for name, addr, h, w, *_ in sensor_groups:
            if h * w < 20:      # 3x3 tips: too coarse for a contact centre
                continue
            m = palm_physical(tactile[name]) if name == "Palm" else tactile[name]
            out[name] = self.surface(m)
        return out

    def update(self, act, tactile, state, since, motors_live):
        if state != "HOLD":
            if self.watching or self.status:
                self.reset()
            return
        now = time.time()
        if not self.watching:
            if motors_live:
                self.angles = [(t, a) for t, a in self.angles if now - t <= SLIP_SETTLE_S]
                self.angles.append((now, list(act["angle_act"])))
            settled = (motors_live and now - self.angles[0][0] >= SLIP_SETTLE_S - 0.05 and
                       all(max(a[i] for _, a in self.angles) - min(a[i] for _, a in self.angles)
                           <= SLIP_SETTLE_TOL for i in range(5)))
            if not settled and now - since < SLIP_SETTLE_MAX_S:
                self.status = "grip settling..."
                return
            self.watching = True
            self.ref_angles = list(act["angle_act"]) if motors_live else None
        if self.reasons:        # latched
            return
        cur = self.surfaces(tactile)
        self.history = [(t, h) for t, h in self.history if now - t <= SLIP_WINDOW_S] + [(now, cur)]
        before = self.history[0][1]
        held = [name for name, (_, p) in cur.items() if p >= SLIP_MIN_PRESSURE]
        self.status = ("watching " + ", ".join(held)) if held else "no contact to watch"
        reasons = []
        for name, (centre0, p0) in before.items():
            centre, p = cur[name]
            if p0 >= SLIP_MIN_PRESSURE and p < SLIP_PRESSURE_DROP * p0:
                reasons.append(f"{name} lost contact" if p == 0 else
                               f"{name} pressure -{100 * (1 - p / p0):.0f}%")
            elif centre0 is not None and centre is not None:
                shift = ((centre[0] - centre0[0]) ** 2 + (centre[1] - centre0[1]) ** 2) ** 0.5
                if shift > SLIP_SHIFT:
                    reasons.append(f"{name} moved {shift:.1f} taxels")
        if motors_live and self.ref_angles is not None:
            for i, key in enumerate(DOF_KEYS[:5]):      # not the thumb rotation
                if act["angle_act"][i] - self.ref_angles[i] > SLIP_ANGLE:     # pushed open
                    reasons.append(f"{key} pushed open {act['angle_act'][i] - self.ref_angles[i]:+d}")
        if reasons:
            self.reasons = reasons
            self.status = "⚠ SLIP: " + "; ".join(reasons)

slip = SlipDetector()


def update_hand_state(act, tactile):
    """Called from the GUI loop with the latest fresh frames (None if stale)."""
    if act is not None:
        step_hand_state(act, tactile)
    update_auto_close(tactile)
    update_auto_record()
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
    if fault and state != "FAULTY" and retry_stall(act, now):
        return
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
        if hand_fully_open(act):
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
    # HOLD stays until Open. Nothing is written in HOLD: the hand's force limit stops the fingers
    # at force_set, and a write while they grip hard can freeze the hand (see write_shorts)

def update_auto_close(tactile):
    """Fires when the palm or a finger pad (AUTO_CLOSE_GROUPS) has contact – only in OPENED, and
    only after all of them have been clear, so the hand doesn't grab again what it has just let
    go of. Auto close then closes the hand, Auto record starts logging."""
    auto_close, auto_rec = auto_hold_var.get(), auto_record_var.get()
    if not (auto_close or auto_rec) or hand["state"] != "OPENED" or not calibrated or tactile is None:
        return
    now = time.time()
    present = auto_close_contact(tactile)
    if not hand["auto_armed"]:
        if present:
            hand["clear_since"] = None
        elif hand["clear_since"] is None:
            hand["clear_since"] = now
        elif now - hand["clear_since"] >= AUTO_HOLD_REARM_S:
            hand["auto_armed"] = True
    elif not present:
        hand["contact_since"] = None
    elif hand["contact_since"] is None:
        hand["contact_since"] = now
    elif now - hand["contact_since"] >= AUTO_HOLD_TRIGGER_S:
        hand.update(auto_armed=False, clear_since=None, contact_since=None)   # once per object
        if auto_rec and not logger.active:
            print(f"Auto record: object detected on {present} – logging started.")
            start_logging()
            hand.update(auto_rec=True, auto_rec_hold_t=None)
        if auto_close:
            print(f"Auto close: object detected on {present}.")
            close_hand()

def update_auto_record():
    """A session Auto record started ends AUTO_RECORD_HOLD_S after the grip is finished (HOLD:
    the index has closed and nothing moves), then the hand opens – also when the motor data has
    been frozen for AUTO_RECORD_FROZEN_S. It also ends when the hand is opened or moved by hand
    before that."""
    if not hand["auto_rec"]:
        return
    if not logger.active:                       # stopped with the button
        hand.update(auto_rec=False, auto_rec_hold_t=None)
        return
    state, now = hand["state"], time.time()
    with latest_lock:
        frozen_for = latest["t"] - latest["act_changed_t"]
    if frozen_for > AUTO_RECORD_FROZEN_S:
        print(f"Auto record: motor data frozen for {frozen_for:.0f} s – logging stopped, opening "
              "anyway. Fingers that stay closed let go once the object is taken out.")
        stop_logging()
        open_hand()
        return
    if state in ("RELEASING", "IDLE"):
        print("Auto record: hand opened / moved before the grip finished – logging stopped.")
        stop_logging()
    elif state == "HOLD":
        if hand["auto_rec_hold_t"] is None:
            hand["auto_rec_hold_t"] = now
        elif now - hand["auto_rec_hold_t"] >= AUTO_RECORD_HOLD_S:
            print(f"Auto record: grip held {AUTO_RECORD_HOLD_S:.0f} s – logging stopped, opening.")
            stop_logging()
            open_hand()

def update_hand_labels():
    state = hand["state"]
    text = f"State: {state}" + (f"  ({hand['reason']})" if hand["reason"] else "")
    hand_state_label.config(text=text, fg=HAND_STATE_COLORS[state])

    if not (auto_hold_var.get() or auto_record_var.get()):
        auto_text = "Off"
    elif not calibrated:
        auto_text = "Waiting for sensor data..."
    elif state != "OPENED":
        auto_text = f"Paused – hand is {state} (runs only when OPENED)"
    elif not hand["auto_armed"]:
        auto_text = "Waiting for the palm and finger pads to be clear..."
    else:
        auto_text = "Armed – place an object on the palm or a finger pad"
    auto_hold_label.config(text=auto_text)
    rot_label.config(text=f"⚠ Thumb rotation stopped: {rot_guard.reason}" if rot_guard.reason else "")
    slip_label.config(text=f"Slip: {slip.status}" if slip.status else "",
                      fg="red" if slip.reasons else "#006600")

def toggle_auto_hold():
    hand.update(clear_since=None, contact_since=None, auto_armed=False)

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

def update_status_label(act, frozen_s=0.0):
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
    if frozen_s:
        lines.insert(0, f"⚠ MOTOR DATA FROZEN for {frozen_s:.1f} s – hand ignores commands "
                        "(remove load / power-cycle)")
        problem = True
    status_label.config(text="\n".join(lines), fg="red" if problem else "black")

# ----------------------------
sliders_initialised = False

def gui_loop():
    global sliders_initialised
    with latest_lock:
        tactile = latest["tactile"]
        act = latest["actuators"]
        t = latest["t"]
        frozen_s = t - latest["act_changed_t"]

    if tactile is not None:
        draw(tactile)

    # don't act on a stale frame if acquisition has stalled, nor on frozen motor feedback
    # (it looks like "nothing moves", which would e.g. trigger HOLD or motors off on old angles)
    fresh = time.time() - t < 0.5
    frozen = act is not None and frozen_s > FREEZE_S
    if frozen and not hand["frozen"]:
        print(f"WARNING: motor feedback frozen (unchanged for {frozen_s:.1f} s) – the hand ignores "
              "commands. Remove the load or power-cycle the hand.")
    elif hand["frozen"] and not frozen:
        print("Motor feedback live again.")
    hand["frozen"] = frozen
    update_hand_state(act if fresh and not frozen else None, tactile if fresh else None)

    # take the current target angles from the hand once at startup
    if not sliders_initialised and act is not None:
        for i in range(6):
            sliders[i].set(act["angle_set"][i])
        sliders_initialised = True

    update_status_label(act if fresh else None, frozen_s if frozen else 0.0)
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

slip_label = tk.Label(status_frame, text="", font=("TkDefaultFont", 11, "bold"), justify="left",
                      wraplength=480)
slip_label.pack(anchor="w")

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
tk.Checkbutton(auto_hold_frame, text="Close hand when an object touches the palm or a finger pad",
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

auto_record_var = tk.BooleanVar(value=False)
tk.Checkbutton(log_frame, text="Auto record: log from object detection until "
               f"{AUTO_RECORD_HOLD_S:g} s after the grip, then open", variable=auto_record_var,
               command=toggle_auto_hold,
               wraplength=300, justify="left").pack(anchor="w")

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
set_speeds()
acq_thread.start()
gui_loop()
root.mainloop()
