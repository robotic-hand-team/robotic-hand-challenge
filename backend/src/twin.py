"""Digital twin geometry and kinematics, taken from the MuJoCo model in simulation/hand_sim.

The frontend gets the visual meshes once (twin.glb, one node per body, in body-local coordinates)
and then only body poses per frame, so hand and object always share the simulation's world frame.
"""
import os
import sys
import threading

import mujoco
import numpy as np
import trimesh

from . import config as C

os.environ["HAND_SIDE"] = C.HAND_SIDE  # read by hand_sim.config at import time
sys.path.insert(0, str(C.SIM_DIR))
from hand_sim import config as SC  # noqa: E402
from hand_sim.scene import build_model  # noqa: E402

VISUAL_GROUP = 1
# Material roles for the frontend (it restyles them in the dashboard theme).
MATERIAL_ROLES = {0: "hand_light", 1: "hand_dark"}


class Twin:
    def __init__(self):
        self.m = build_model()
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.dof_qadr = np.array([m.jnt_qposadr[m.joint(j).id] for j in SC.DOF_JOINTS])
        self.dof_qmax = np.array([m.jnt_range[m.joint(j).id][1] for j in SC.DOF_JOINTS])
        # Coupled finger joints (equality constraints of type JOINT): q1 = poly(q2).
        self.couplings = [(m.jnt_qposadr[m.eq_obj1id[e]], m.jnt_qposadr[m.eq_obj2id[e]], m.eq_data[e][:5].copy())
                          for e in range(m.neq) if m.eq_type[e] == mujoco.mjtEq.mjEQ_JOINT]
        self.hand_bodies = [m.body(b).name for b in range(m.nbody)
                            if m.body(b).name.startswith(f"{C.HAND_SIDE}_")]
        self.objects = list(SC.OBJECTS)
        self.obj_qadr = {o: m.jnt_qposadr[m.joint(f"{o}_free").id] for o in self.objects}
        self.root = f"{C.HAND_SIDE}_hand_root"
        self.lock = threading.Lock()

    def export_glb(self, path):
        """Write every visual mesh as one node per body (body-local coordinates, metres)."""
        m = self.m
        scene = trimesh.Scene()
        for g in range(m.ngeom):
            if m.geom_group[g] != VISUAL_GROUP or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            body = m.body(m.geom_bodyid[g]).name
            mid = m.geom_dataid[g]
            va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, m.geom_quat[g])
            verts = m.mesh_vert[va:va + vn] @ R.reshape(3, 3).T + m.geom_pos[g]
            mesh = trimesh.Trimesh(verts, m.mesh_face[fa:fa + fn], process=False)
            if len(mesh.faces) > 60000:
                mesh = mesh.simplify_quadric_decimation(face_count=60000)
            role = body if body in self.objects else MATERIAL_ROLES.get(int(m.geom_matid[g]), "hand_light")
            mesh.visual = trimesh.visual.TextureVisuals(
                material=trimesh.visual.material.PBRMaterial(name=role, baseColorFactor=m.geom_rgba[g]))
            scene.add_geometry(mesh, node_name=body, geom_name=f"{body}_g{g}")
        path.parent.mkdir(parents=True, exist_ok=True)
        scene.export(path)

    def poses(self, angles: np.ndarray, obj: str | None = None, obj_pose: np.ndarray | None = None) -> dict:
        """Body poses [x, y, z, qw, qx, qy, qz] for hand angles (0-1000, DOF order) and, optionally,
        one object's pose [x, y, z, qw, qx, qy, qz]."""
        with self.lock:  # replay requests (thread pool) and live frames (event loop) share one MjData
            d = self.d
            d.qpos[:] = 0
            d.qpos[self.dof_qadr] = self.dof_qmax * (1.0 - np.clip(angles, 0, 1000) / 1000.0)
            for q1, q2, c in self.couplings:
                x = d.qpos[q2]
                d.qpos[q1] = c[0] + c[1] * x + c[2] * x**2 + c[3] * x**3 + c[4] * x**4
            names = list(self.hand_bodies)
            if obj is not None:
                a = self.obj_qadr[obj]
                d.qpos[a:a + 7] = obj_pose
                names.append(obj)
            mujoco.mj_kinematics(self.m, d)
            out = {}
            for name in names:
                b = self.m.body(name).id
                out[name] = np.round(np.concatenate([d.xpos[b], d.xquat[b]]), 5).tolist()
            return out

    def root_pos(self):
        return self.m.body(self.root).pos.copy()
