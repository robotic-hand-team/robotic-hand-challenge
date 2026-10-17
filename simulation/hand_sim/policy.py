"""Grasp policies. A policy sees the same registers the real hand exposes and writes ANGLE/FORCE/SPEED_SET.

To plug in the neural network, subclass Policy (see NNPolicy) and pass it to HandoverEpisode.
"""

from dataclasses import dataclass, field

import numpy as np

from . import config as C

# Episode phases, in order.
PHASE_APPROACH = "approach"  # carrier moves the object into the open hand
PHASE_GRASP = "grasp"  # object held still by the carrier, policy closes the hand
PHASE_HOLD = "hold"  # carrier released, hand alone holds the object
PHASES = (PHASE_APPROACH, PHASE_GRASP, PHASE_HOLD)


@dataclass
class Observation:
    t: float  # seconds since episode start
    phase: str
    phase_t: float  # seconds since the current phase started
    registers: dict  # HandController.read(): angle_set/angle_act/force_act/current/error/status (DOF order)
    tactile: dict | None = None  # filled in by the tactile model (milestone 3)
    proximity: float = 1.0  # distance (m) from object to palm body


@dataclass
class Action:
    angle_set: list[int] | None = None  # None = leave registers unchanged; -1 per DOF = no action
    force_set: list[int] | None = None
    speed_set: list[int] | None = None
    release: bool = False  # ask the carrier to let go (only honoured in the grasp phase)
    info: dict = field(default_factory=dict)


class Policy:
    def reset(self, episode_info: dict) -> None:
        pass

    def act(self, obs: Observation) -> Action:
        raise NotImplementedError


class ScriptedGraspPolicy(Policy):
    """Pre-shape during the approach, close everything with a force limit, release once the fingers stop."""

    CLOSE_ANGLES = [0, 0, 0, 0, 0, -1]  # thumb rotation stays opposed from the pre-shape
    SETTLE_S = 0.2  # fingers must be stationary this long before releasing

    def __init__(self, force_g: int | None = None):
        self.force_g = force_g
        self._last_angles = None
        self._still_since = None

    def reset(self, episode_info):
        self._force = self.force_g or C.OBJECTS[episode_info["object"]]["grasp_force_g"]
        self._last_angles = None
        self._still_since = None
        self._closing_sent = False

    def act(self, obs):
        if obs.phase == PHASE_APPROACH:
            return Action(angle_set=C.PRESHAPE_ANGLES, force_set=[self._force] * 4 + [1000, 200])
        if obs.phase == PHASE_GRASP:
            if not self._closing_sent:
                self._closing_sent = True
                return Action(angle_set=self.CLOSE_ANGLES, force_set=[self._force] * 4 + [1000, 200])
            angles = np.asarray(obs.registers["angle_act"])
            if self._last_angles is not None and np.abs(angles - self._last_angles).max() <= 2:
                self._still_since = obs.t if self._still_since is None else self._still_since
            else:
                self._still_since = None
            self._last_angles = angles
            settled = self._still_since is not None and obs.t - self._still_since >= self.SETTLE_S
            return Action(release=settled)
        return Action()


class NNPolicy(Policy):
    """Placeholder for the neural-network policy (models/checkpoints/ is gitignored)."""

    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path

    def act(self, obs):
        raise NotImplementedError(f"NN policy not implemented yet (checkpoint: {self.checkpoint_path})")
