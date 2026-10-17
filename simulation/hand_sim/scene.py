"""Build the MuJoCo model: Inspire right hand + grasp objects carried by mocap bodies."""

import json

import mujoco
import numpy as np
import trimesh

from . import config as C
from .left_hand import mirror_spec

SCENE_XML = C.SIM_DIR / "Inspire_hand" / "xml" / "scene.xml"

# Joint-space position servo emulating the hand's linear actuators (non-backdrivable, stiff).
SERVO_KP = 20.0  # Nm/rad
JOINT_ARMATURE = 0.002  # reflected actuator inertia, also keeps the stiff servo stable at dt=2 ms
JOINT_DAMPING = 0.02
# Torque limit = max fingertip force (FORCE_SET_MAX grams) times a nominal lever arm.
NOMINAL_LEVER_M = 0.06

OBJECT_FRICTION = [1.0, 0.005, 0.0001]
# Soft weld = compliant handover (like a human hand or a compliant arm): closing fingers can pull the
# object into the palm. Gravity is compensated separately while carried (World.place), so no sag.
CARRIER_WELD_TIMECONST = 0.1
PARK_POS = {"vegabar_29": [0.5, -0.3, 0.15], "vegabar_38": [0.5, 0.3, 0.15]}


def _hand_bodies(spec):
    return [b for b in spec.bodies if b.name.startswith(f"{C.HAND_SIDE}_")]


def _configure_hand(spec):
    for body in _hand_bodies(spec):
        for geom in body.geoms:
            if geom.group == 2:
                geom.contype, geom.conaffinity = C.HAND_CONTYPE, C.HAND_CONAFFINITY
                geom.condim = 4
                geom.friction = OBJECT_FRICTION
    for joint in spec.joints:
        if joint.name.startswith(f"{C.HAND_SIDE}_"):
            joint.actfrclimited = False
            joint.armature = JOINT_ARMATURE
            joint.damping = [JOINT_DAMPING, 0.0, 0.0]

    for act in list(spec.actuators):
        spec.delete(act)
    max_torque = C.FORCE_SET_MAX * 9.81e-3 * NOMINAL_LEVER_M
    for name, jname in zip(C.DOF_NAMES, C.DOF_JOINTS):
        joint = spec.joint(jname)
        act = spec.add_actuator(name=f"act_{name}", target=jname, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        act.set_to_position(kp=SERVO_KP, kv=2 * np.sqrt(SERVO_KP * JOINT_ARMATURE))
        act.ctrllimited = True
        act.ctrlrange = list(joint.range)
        act.forcelimited = True
        act.forcerange = [-max_torque, max_torque]


def _add_object(spec, obj_id, pos):
    cfg = C.OBJECTS[obj_id]
    obj_dir = C.OBJECTS_DIR / obj_id
    meta = json.loads((obj_dir / "object.json").read_text())

    volumes = [trimesh.load(obj_dir / f, force="mesh").volume for f in meta["collision_parts"]]
    density = cfg["mass_kg"] / sum(volumes)

    spec.add_mesh(name=f"{obj_id}_visual", file=str(obj_dir / "visual.obj"))
    body = spec.worldbody.add_body(name=obj_id, pos=pos)
    body.add_freejoint(name=f"{obj_id}_free")
    body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=f"{obj_id}_visual",
        rgba=cfg["rgba"],
        contype=0,
        conaffinity=0,
        group=1,
        mass=0,
    )
    for i, part in enumerate(meta["collision_parts"]):
        mesh_name = f"{obj_id}_col{i:02d}"
        spec.add_mesh(name=mesh_name, file=str(obj_dir / part), maxhullvert=64)
        body.add_geom(
            name=mesh_name,
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=mesh_name,
            contype=C.OBJECT_CONTYPE,
            conaffinity=C.OBJECT_CONAFFINITY,
            condim=4,
            friction=OBJECT_FRICTION,
            density=density,
            group=3,
            rgba=[0.9, 0.4, 0.1, 0.5],
        )

    spec.worldbody.add_body(name=f"{obj_id}_carrier", mocap=True, pos=pos)
    # data = anchor(3), relpose pos(3), relpose quat(4), torquescale. Set explicitly: MjSpec's defaults
    # (anchor 0 1 0, relpos 0 1 0) only behave like an identity weld when both frames are unrotated.
    spec.add_equality(
        name=f"{obj_id}_weld",
        type=mujoco.mjtEq.mjEQ_WELD,
        objtype=mujoco.mjtObj.mjOBJ_BODY,
        name1=obj_id,
        name2=f"{obj_id}_carrier",
        data=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        solref=[CARRIER_WELD_TIMECONST, 1.0],
    )


def build_model() -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(SCENE_XML))
    if C.HAND_SIDE == "left":
        mirror_spec(spec, SCENE_XML.parent)
    spec.option.timestep = 0.002
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    _configure_hand(spec)
    for obj_id in C.OBJECTS:
        _add_object(spec, obj_id, PARK_POS[obj_id])
    return spec.compile()
