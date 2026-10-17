"""Build the MuJoCo model: Inspire right hand + grasp objects carried by mocap bodies."""

import json

import mujoco
import numpy as np
import trimesh

from . import config as C

SCENE_XML = C.SIM_DIR / "Inspire_hand" / "xml" / "scene.xml"

# Joint-space position servo emulating the hand's linear actuators (non-backdrivable, stiff).
SERVO_KP = 200.0  # Nm/rad (fingers not moved by object contact)
JOINT_ARMATURE = 0.002  # reflected actuator inertia; also what keeps the stiff servo stable at TIMESTEP
JOINT_DAMPING = 0.02
# Torque limit = max fingertip force (FORCE_SET_MAX grams) times a nominal lever arm.
NOMINAL_LEVER_M = 0.06

OBJECT_FRICTION = [1.0, 0.005, 0.0001]
# Soft weld = compliant handover (like a human hand or a compliant arm): closing fingers can pull the
# object into the palm. Gravity is compensated separately while carried (World.place), so no sag.
CARRIER_WELD_TIMECONST = 0.1
PARK_POS = {"vegabar_29": [0.5, -0.3, 0.15], "vegabar_38": [0.5, 0.3, 0.15]}


def _hand_bodies(spec):
    return [b for b in spec.bodies if b.name.startswith("right_")]


def _configure_hand(spec):
    for body in _hand_bodies(spec):
        for geom in body.geoms:
            if geom.group == 2:
                geom.contype, geom.conaffinity = C.HAND_CONTYPE, C.HAND_CONAFFINITY
                geom.condim = 4
                geom.friction = OBJECT_FRICTION
                if C.CONTACT_SOLREF is not None:
                    geom.solref = list(C.CONTACT_SOLREF)
                if C.CONTACT_SOLIMP is not None:
                    geom.solimp = list(C.CONTACT_SOLIMP)
    for joint in spec.joints:
        if joint.name.startswith("right_"):
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


REBUILT_COLLISION_DIR = C.SIM_DIR / "Inspire_hand" / "collision_rebuilt"
COLLISION_MANIFEST = REBUILT_COLLISION_DIR / "manifest.json"


def _rebuilt_collision_parts() -> dict[str, list[str]]:
    """mesh asset name -> rebuilt collision part files, or {} when the rebuild has not been run."""
    if not COLLISION_MANIFEST.exists():
        return {}
    manifest = json.loads(COLLISION_MANIFEST.read_text())
    return {name: [str(REBUILT_COLLISION_DIR / p) for p in entry["parts"]]
            for name, entry in manifest.items()}


def _swap_collision_geometry(spec):
    """Replace the vendor collision geometry with the rebuilt convex decompositions.

    See rebuild_hand_collision.py for why: the vendor collision geometry bulges 1.6-7.2 mm past the
    drawn surface on the fingers and 18.3 mm on the palm, so contacts fire early, the fingers block
    each other before they visually touch, and the tactile patches sit on the inflated shape. With no
    manifest present this is a no-op and the vendor geometry is used exactly as before, so deleting
    collision_rebuilt/ restores the previous behaviour.

    A manifest key names what is being replaced, because the two cases are structurally different:

    * a collision *mesh* asset name, when the link collides as a mesh. The vendor shares one mesh
      across all four fingers (middle/ring/pinky_proximal all point at right_index_proximal.obj, and
      the drawn meshes are shared the same way), so the parts are added once and every holder body
      gets its own geoms.
    * a *body* name, when the link collides as primitives - the palm is 7 boxes and a cylinder.

    Either way every group-2 geom on the body is replaced, the drawn geoms are untouched so the hand
    renders identically, and tactile.py (which derives each sensor patch from the group-2 geoms'
    vertices) ends up on the true surface.
    """
    parts = _rebuilt_collision_parts()
    if not parts:
        return
    bodies = _hand_bodies(spec)
    for key, files in parts.items():
        mesh_holders = [b for b in bodies
                        if any(g.group == 2 and g.meshname == key for g in b.geoms)]
        targets = mesh_holders or [b for b in bodies
                                   if b.name == key and any(g.group == 2 for g in b.geoms)]
        if not targets:
            continue
        part_names = []
        for i, path in enumerate(files):
            part_name = f"{key}_rb{i:02d}"
            spec.add_mesh(name=part_name, file=path, maxhullvert=C.MAX_HULL_VERT)
            part_names.append(part_name)
        for body in targets:
            for geom in [g for g in body.geoms if g.group == 2]:
                spec.delete(geom)
            for part_name in part_names:
                body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=part_name, group=2)


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
        spec.add_mesh(name=mesh_name, file=str(obj_dir / part), maxhullvert=C.MAX_HULL_VERT)
        geom = body.add_geom(
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
        if C.CONTACT_SOLREF is not None:
            geom.solref = list(C.CONTACT_SOLREF)
        if C.CONTACT_SOLIMP is not None:
            geom.solimp = list(C.CONTACT_SOLIMP)

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


def _exclude_adjacent_hand_bodies(spec):
    """Exclude contacts between each hand body and its parent.

    MuJoCo's option.filterparent is meant to do this and is enabled here, but this build still
    emits contacts between the palm and every finger's proximal segment, overlapping by 11-13 mm:
    the finger roots physically sit inside the palm volume. Resolved as real contacts they produce
    forces in the hundreds of kilonewtons and the grasp explodes (measured: 2.7 MN
    palm<->thumb_proximal_base). Those overlaps are an artifact of the link geometry, not a grasp
    event, so they are excluded explicitly. Contacts between different fingers - the ones that
    actually stop fingers passing through each other - are unaffected.
    """
    hand_names = {b.name for b in _hand_bodies(spec)}
    for body in _hand_bodies(spec):
        parent = body.parent
        if parent is not None and parent.name in hand_names:
            spec.add_exclude(bodyname1=body.name, bodyname2=parent.name)


def _configure_floor(spec):
    """Put the ground plane on its own collision bit so the hand cannot hit it (see config)."""
    for geom in spec.worldbody.geoms:
        geom.contype, geom.conaffinity = C.FLOOR_CONTYPE, C.FLOOR_CONAFFINITY
        if C.CONTACT_SOLREF is not None:
            geom.solref = list(C.CONTACT_SOLREF)
        if C.CONTACT_SOLIMP is not None:
            geom.solimp = list(C.CONTACT_SOLIMP)


def build_model() -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(SCENE_XML))
    spec.option.timestep = C.TIMESTEP
    spec.option.iterations = C.SOLVER_ITERATIONS
    spec.option.ls_iterations = C.LS_ITERATIONS
    spec.option.noslip_iterations = C.NOSLIP_ITERATIONS
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    _swap_collision_geometry(spec)
    _configure_hand(spec)
    if C.HAND_ALLOWS_SELF_COLLISION:
        _exclude_adjacent_hand_bodies(spec)
    _configure_floor(spec)
    for obj_id in C.OBJECTS:
        _add_object(spec, obj_id, PARK_POS[obj_id])
    return spec.compile()
