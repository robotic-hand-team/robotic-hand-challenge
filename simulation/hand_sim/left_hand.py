"""Build the LEFT Inspire hand from the right-hand MuJoCo model by mirroring it.

The Inspire left hand is the mirror image of the right one: the left URDF (pose/inspire_hand) has the
same links, joints and limits, with the thumb on the other side. Here the right model is reflected
across the plane through the hand root that separates thumb side and little-finger side (sim y -> -y,
S = diag(1, -1, 1)):

  body / geom / joint / inertial positions:  p -> S p
  orientations:                              R -> S R S       (quaternion w,x,y,z -> w,-x,y,-z)
  hinge axes:                                a -> -S a         (S R(a, t) S = R(-S a, t): same joint
                                                                  angle = same flexion, mirrored)
  mesh vertices:                             v -> S v, faces re-wound so normals still point out
  names:                                     right_* -> left_*

Checked against pose/inspire_hand/inspire_hand_left.urdf by test_left_hand() below.
"""
from pathlib import Path

import mujoco
import numpy as np
import trimesh

S = np.diag([1.0, -1.0, 1.0])


def _mq(q):
    w, x, y, z = q
    return [w, -x, y, -z]


def _mp(p):
    return [p[0], -p[1], p[2]]


def mirror_spec(spec: mujoco.MjSpec, xml_dir: Path, src: str = "right", dst: str = "left") -> None:
    """Mirror every `src_*` element of the spec in place and rename it `dst_*`."""
    root = next(b for b in spec.bodies if b.name == f"{src}_hand_root")
    hand_bodies = [b for b in spec.bodies if b.name.startswith(f"{src}_")]
    for b in hand_bodies:
        if b is not root:  # the root keeps its world pose; everything in its frame is mirrored
            b.pos, b.quat = _mp(b.pos), _mq(b.quat)
        b.ipos, b.iquat = _mp(b.ipos), _mq(b.iquat)
        for g in b.geoms:
            g.pos, g.quat = _mp(g.pos), _mq(g.quat)
            if g.name.startswith(f"{src}_"):
                g.name = dst + g.name[len(src):]
            if g.meshname.startswith(f"{src}_"):
                g.meshname = dst + g.meshname[len(src):]
        for j in b.joints:
            j.pos = _mp(j.pos)
            j.axis = list(-(S @ np.asarray(j.axis, float)))
            j.name = dst + j.name[len(src):]
        b.name = dst + b.name[len(src):]

    for m in spec.meshes:
        if not m.name.startswith(f"{src}_"):
            continue
        mesh = trimesh.load(xml_dir / m.file, force="mesh", process=False)
        v = np.asarray(mesh.vertices, float) * np.asarray(m.scale, float)
        v[:, 1] *= -1
        m.file = ""
        m.scale = [1.0, 1.0, 1.0]
        m.uservert = v.ravel().tolist()
        m.userface = np.asarray(mesh.faces)[:, ::-1].ravel().tolist()  # reflection flips winding
        m.name = dst + m.name[len(src):]

    for e in spec.equalities:
        e.name1 = e.name1.replace(f"{src}_", f"{dst}_", 1)
        e.name2 = e.name2.replace(f"{src}_", f"{dst}_", 1)
    for a in list(spec.actuators):  # hand_sim.scene replaces the actuators anyway
        spec.delete(a)


def test_left_hand(n_poses: int = 5, seed: int = 0) -> float:
    """Max distance (m) between the mirrored model's link origins and the left URDF's, over random
    joint angles. Run from simulation/:  python -m hand_sim.left_hand"""
    import yourdfpy
    from scipy.spatial.transform import Rotation

    sim_dir = Path(__file__).resolve().parents[1]
    xml = sim_dir / "Inspire_hand" / "xml" / "scene.xml"
    spec = mujoco.MjSpec.from_file(str(xml))
    mirror_spec(spec, xml.parent)
    m = spec.compile()
    d = mujoco.MjData(m)
    urdf = yourdfpy.URDF.load(str(sim_dir.parent / "pose" / "inspire_hand" / "inspire_hand_left.urdf"))
    R_base_hbl = Rotation.from_euler("xyz", [-1.5707963, 0, 3.14159]).as_matrix()
    root = m.body("left_hand_root").id
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n_poses):
        cfg = {}
        for jn in urdf.actuated_joint_names:
            lim = urdf.joint_map[jn].limit
            cfg[jn] = rng.uniform(lim.lower, lim.upper)
        urdf.update_cfg(cfg)
        mujoco.mj_resetData(m, d)
        for jn, q in zip(urdf.actuated_joint_names, urdf.cfg):
            d.qpos[m.jnt_qposadr[m.joint(f"left_{jn}").id]] = q
        for e in range(m.neq):  # coupled joints, as in the URDF mimic tags
            j1, j2 = m.eq_obj1id[e], m.eq_obj2id[e]
            c = m.eq_data[e][:5]
            q2 = d.qpos[m.jnt_qposadr[j2]]
            d.qpos[m.jnt_qposadr[j1]] = c[0] + c[1] * q2
        mujoco.mj_forward(m, d)
        for link in ["thumb_distal", "index_intermediate", "pinky_intermediate", "middle_proximal"]:
            p_urdf = urdf.get_transform(link, "base")[:3, 3]          # URDF 'base' = sim hand root frame
            p_sim = d.xmat[root].reshape(3, 3).T @ (d.xpos[m.body(f"left_{link}").id] - d.xpos[root])
            worst = max(worst, float(np.linalg.norm(p_urdf - p_sim)))
    return worst


if __name__ == "__main__":
    print(f"left hand vs left URDF: max link position error {test_left_hand() * 1000:.3f} mm")
