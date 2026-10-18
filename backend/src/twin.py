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
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

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
        # live grasp: visual geoms of each finger (DOF order, thumb = its bending segments) and each
        # object's collision hulls, for stopping fingers at the object surface (clamp_to_object)
        def visual(word):
            return [g for g in range(m.ngeom) if m.geom_group[g] == VISUAL_GROUP
                    and word in m.body(m.geom_bodyid[g]).name
                    and m.body(m.geom_bodyid[g]).name.startswith(f"{C.HAND_SIDE}_")
                    and "proximal_base" not in m.body(m.geom_bodyid[g]).name]
        self.finger_geoms = [visual(w) for w in ("pinky", "ring", "middle", "index", "thumb")]
        self.obj_hulls = {o: [g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body(o).id
                              and m.geom_group[g] == 3] for o in self.objects}
        self._clamp_cache = {}
        self._fromto = np.zeros(6)
        # fast inside test for clamp_to_object: points on each finger's drawn surface (geom frame,
        # at most PROBE_POINTS per geom) and each object hull's planes (object body frame)
        PROBE_POINTS = 400
        def geom_points(g):
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, m.geom_quat[g])
            v = v @ R.reshape(3, 3).T + m.geom_pos[g]          # mesh frame -> geom's body frame
            return v[::max(1, len(v) // PROBE_POINTS)]
        self.finger_points = [[(m.geom_bodyid[g], geom_points(g)) for g in geoms]
                              for geoms in self.finger_geoms]
        self.obj_planes = {o: [ConvexHull(geom_points(g)).equations for g in hulls]
                           for o, hulls in self.obj_hulls.items()}
        # drawn palm surface (palm faces +x): its vertices, for placing an object against it
        self._set_qpos(np.full(6, 1000.0))
        root_id = m.body(self.root).id
        self.palm_verts = np.vstack([
            m.mesh_vert[m.mesh_vertadr[m.geom_dataid[g]]:m.mesh_vertadr[m.geom_dataid[g]] + m.mesh_vertnum[m.geom_dataid[g]]]
            @ self.d.geom_xmat[g].reshape(3, 3).T + self.d.geom_xpos[g]
            for g in range(m.ngeom) if m.geom_bodyid[g] == root_id and m.geom_group[g] == VISUAL_GROUP])
        self._cell = 0.002
        pv = self.palm_verts
        self._grid_min = pv[:, 1:].min(axis=0)
        idx = np.floor((pv[:, 1:] - self._grid_min) / self._cell).astype(int)
        grid = np.full(idx.max(axis=0) + 1, -np.inf)
        np.maximum.at(grid, (idx[:, 0], idx[:, 1]), pv[:, 0])
        k = 2                                       # +-2 cells = the 4 mm neighbourhood
        padded = np.pad(grid, k, constant_values=-np.inf)
        self._palm_grid = np.max([padded[k + a:k + a + grid.shape[0], k + b:k + b + grid.shape[1]]
                                  for a in range(-k, k + 1) for b in range(-k, k + 1)], axis=0)

    def _set_qpos(self, angles, obj=None, obj_pose=None):
        """qpos for hand angles (0-1000, DOF order) incl. the coupled joints, optionally one object's
        pose; then forward kinematics. Caller holds self.lock (or is single-threaded)."""
        d = self.d
        d.qpos[:] = 0
        d.qpos[self.dof_qadr] = self.dof_qmax * (1.0 - np.clip(angles, 0, 1000) / 1000.0)
        for q1, q2, c in self.couplings:
            x = d.qpos[q2]
            d.qpos[q1] = c[0] + c[1] * x + c[2] * x**2 + c[3] * x**3 + c[4] * x**4
        if obj is not None:
            a = self.obj_qadr[obj]
            d.qpos[a:a + 7] = obj_pose
        mujoco.mj_kinematics(self.m, d)

    def _finger_distance(self, i, obj):
        """Smallest signed distance (m) between finger i's drawn segments and the object's hulls
        (< 0: the finger is inside the object). Exact but slow when deep inside: only for the
        one-off fit_real_grasp; clamp_to_object uses _finger_inside."""
        return min(mujoco.mj_geomDistance(self.m, self.d, g1, g2, 0.05, self._fromto)
                   for g1 in self.finger_geoms[i] for g2 in self.obj_hulls[obj])

    def _finger_inside(self, i, obj, tol=0.0005):
        """True if any sampled point of finger i's drawn surface is inside one of the object's convex
        hulls (deeper than tol). Vectorised plane tests: fast even when deep inside."""
        d = self.d
        ob = self.m.body(obj).id
        Ro, po = d.xmat[ob].reshape(3, 3), d.xpos[ob]
        for body, pts in self.finger_points[i]:
            world = pts @ d.xmat[body].reshape(3, 3).T + d.xpos[body]
            local = (world - po) @ Ro                          # world -> object body frame
            for eq in self.obj_planes[obj]:
                if np.any(np.all(local @ eq[:, :3].T + eq[:, 3] < -tol, axis=1)):
                    return True
        return False

    def clamp_to_object(self, angles, obj, obj_pose, iters=7):
        """Hand angles with every finger and the thumb bend opened just enough not to pass into the
        object (the real fingers can't either). Caller holds self.lock."""
        key = (obj, tuple(np.round(angles[:6] / 10).astype(int)), tuple(np.round(obj_pose, 4)))
        if key in self._clamp_cache:
            return self._clamp_cache[key]
        out = np.array(angles, float)
        # the thumb only moves if opening it can free it at all; otherwise it stays where it really is
        thumb_free = out.copy()
        thumb_free[4:6] = 1000.0
        self._set_qpos(thumb_free, obj, obj_pose)
        fingers = range(5) if not self._finger_inside(4, obj) else range(4)
        for i in fingers:
            self._set_qpos(out, obj, obj_pose)
            if not self._finger_inside(i, obj):
                continue
            closed, opened = out[i], 1000.0          # inside at `closed`; find the least opening
            for _ in range(iters):
                mid = (closed + opened) / 2
                out[i] = mid
                self._set_qpos(out, obj, obj_pose)
                if not self._finger_inside(i, obj):
                    opened = mid
                else:
                    closed = mid
            out[i] = opened
        self._set_qpos(out, obj, obj_pose)
        if 4 in fingers and self._finger_inside(4, obj):
            # thumb still inside with its bend open: turn the rotation out (towards 1000) as well
            closed, opened = out[5], 1000.0
            for _ in range(iters):
                mid = (closed + opened) / 2
                out[5] = mid
                self._set_qpos(out, obj, obj_pose)
                if not self._finger_inside(4, obj):
                    opened = mid
                else:
                    closed = mid
            out[5] = opened
        if len(self._clamp_cache) > 256:
            self._clamp_cache.clear()
        self._clamp_cache[key] = out
        return out

    def _palm_front(self, yz):
        """Front (max x) of the drawn palm at points yz [n, 2]; -inf off the palm."""
        idx = np.floor((np.asarray(yz) - self._grid_min) / self._cell).astype(int)
        ok = np.all((idx >= 0) & (idx < self._palm_grid.shape), axis=1)
        out = np.full(len(idx), -np.inf)
        out[ok] = self._palm_grid[idx[ok, 0], idx[ok, 1]]
        return out

    def fit_real_grasp(self, obj, hold_angles, gap=0.001):
        """Object pose [x, y, z, qw, qx, qy, qz] for the real (left) hand's grasp: nominal orientation
        and y (long axis across the fingers), x so it rests `gap` off the drawn palm, and the height z
        at which the fingers, at the real hold angles, just touch it (sum of |finger distance| least)."""
        quat = np.array(SC.OBJECTS[obj]["grasp_quat"], float)
        y = SC.OBJECTS[obj]["grasp_pos"][1]
        hold = np.array(hold_angles, float)
        best = None
        with self.lock:
            for z in np.arange(0.05, 0.13, 0.0025):
                self._set_qpos(np.full(6, 1000.0), obj, np.array([0.0, y, z, *quat]))
                verts = np.vstack([
                    self.m.mesh_vert[self.m.mesh_vertadr[self.m.geom_dataid[g]]:self.m.mesh_vertadr[self.m.geom_dataid[g]]
                                     + self.m.mesh_vertnum[self.m.geom_dataid[g]]]
                    @ self.d.geom_xmat[g].reshape(3, 3).T + self.d.geom_xpos[g] for g in self.obj_hulls[obj]])
                margin = np.min(verts[:, 0] - self._palm_front(verts[:, 1:]))
                pose = np.array([gap - margin, y, z, *quat])
                self._set_qpos(hold, obj, pose)
                miss = sum(abs(self._finger_distance(i, obj)) for i in range(4))
                if best is None or miss < best[0]:
                    best = (miss, pose)
        return best[1]

    def pose_in_palm(self, obj, theta_deg, nominal_pose, nominal_theta_deg, grip, fist, gap=0.001):
        """Object pose [x, y, z, qw, qx, qy, qz] with its yellow end (object +y) pointing theta_deg in the
        palm plane (0 = towards the thumb, +y; 90 = towards the fingertips, +z): nominal_pose's orientation
        (whose yellow end points nominal_theta_deg) turned about the palm normal (x), the object's grip
        point (object frame, m) on the fist centre (y, z), and x so it rests `gap` off the drawn palm."""
        w, qx, qy, qz = nominal_pose[3:]
        R = (Rotation.from_euler("x", theta_deg - nominal_theta_deg, degrees=True)
             * Rotation.from_quat([qx, qy, qz, w]))
        y, z = np.asarray(fist, float) - R.apply(grip)[1:]
        qx, qy, qz, w = R.as_quat()
        quat = np.array([w, qx, qy, qz])
        with self.lock:
            self._set_qpos(np.full(6, 1000.0), obj, np.array([0.0, y, z, *quat]))
            verts = np.vstack([
                self.m.mesh_vert[self.m.mesh_vertadr[self.m.geom_dataid[g]]:self.m.mesh_vertadr[self.m.geom_dataid[g]]
                                 + self.m.mesh_vertnum[self.m.geom_dataid[g]]]
                @ self.d.geom_xmat[g].reshape(3, 3).T + self.d.geom_xpos[g] for g in self.obj_hulls[obj]])
        margin = np.min(verts[:, 0] - self._palm_front(verts[:, 1:]))
        x = gap - margin if np.isfinite(margin) else nominal_pose[0]   # nothing over the palm: keep nominal x
        return np.array([x, y, z, *quat])

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

    def poses(self, angles: np.ndarray, obj: str | None = None, obj_pose: np.ndarray | None = None,
              clamp: bool = False) -> dict:
        """Body poses [x, y, z, qw, qx, qy, qz] for hand angles (0-1000, DOF order) and, optionally,
        one object's pose [x, y, z, qw, qx, qy, qz]. clamp: fingers stop at the object's surface."""
        with self.lock:  # replay requests (thread pool) and live frames (event loop) share one MjData
            d = self.d
            if obj is not None and clamp:
                angles = self.clamp_to_object(np.asarray(angles, float), obj, obj_pose)
            self._set_qpos(angles, obj, obj_pose)
            names = list(self.hand_bodies) + ([obj] if obj is not None else [])
            out = {}
            for name in names:
                b = self.m.body(name).id
                out[name] = np.round(np.concatenate([d.xpos[b], d.xquat[b]]), 5).tolist()
            return out

    def root_pos(self):
        return self.m.body(self.root).pos.copy()
