"""Inspire hand kinematics and taxel-to-3D mapping.

All 3D results are expressed in the `hand_base_link` frame, in metres.
"""
import numpy as np
import trimesh
import yourdfpy

import config as C

REF = "hand_base_link"


class HandModel:
    def __init__(self, urdf_path=None):
        urdf_path = urdf_path or C.HAND_URDF  # read at call time, so config.set_hand() takes effect
        self.urdf = yourdfpy.URDF.load(str(urdf_path), load_meshes=False,
                                       build_scene_graph=True)
        self.dir = urdf_path.parent
        self.link_mesh = {name: self._load_link_mesh(name) for name in self.urdf.link_map}
        self.link_mesh = {k: v for k, v in self.link_mesh.items() if v is not None}
        self._taxel_cache = {}

    # ---- kinematics ------------------------------------------------------
    def set_raw(self, raw: dict):
        """Set joints from raw actuator values (0..1000, 1000 = open)."""
        cfg = {}
        for dof, joint in C.DOF_TO_JOINT.items():
            upper = self.urdf.joint_map[joint].limit.upper
            frac = np.clip((C.RAW_OPEN - raw[dof]) / C.RAW_OPEN, 0, 1)
            scale = C.THUMB_ANGLE_SCALE if dof.startswith("thumb") else C.FINGER_ANGLE_SCALE
            cfg[joint] = frac * upper * scale
        self.urdf.update_cfg(cfg)

    def T(self, link):
        """4x4 pose of a link in the hand_base_link frame."""
        return np.linalg.inv(self.urdf.get_transform(REF, "base")) @ self.urdf.get_transform(link, "base")

    # ---- meshes ------------------------------------------------------------
    def _load_link_mesh(self, link):
        parts = []
        for vis in self.urdf.link_map[link].visuals:
            if vis.geometry.mesh is None:
                continue
            m = trimesh.load(self.dir / vis.geometry.mesh.filename, force="mesh")
            if vis.origin is not None:
                m.apply_transform(vis.origin)
            parts.append(m)
        return trimesh.util.concatenate(parts) if parts else None

    def posed_meshes(self):
        """Hand link meshes at the current configuration (hand frame)."""
        out = {}
        for link, m in self.link_mesh.items():
            mm = m.copy()
            mm.apply_transform(self.T(link))
            out[link] = mm
        return out

    # ---- taxel geometry ----------------------------------------------------
    def _child_offset(self, link):
        """Vector from this link's origin to its child joint (link frame)."""
        for j in self.urdf.joint_map.values():
            if j.parent == link:
                return j.origin[:3, 3]
        return None

    def _parent_axis(self, link):
        for j in self.urdf.joint_map.values():
            if j.child == link and j.axis is not None:
                return np.asarray(j.axis, float)
        return np.array([0, 0, 1.0])

    def _patch(self, link, rows, cols, span, width):
        """Ray-cast a rows x cols grid onto the palm-facing side of a link.

        Returns points (rows, cols, 3) and normals in the link frame.
        """
        mesh = self.link_mesh[link]
        long_vec = self._child_offset(link)
        if long_vec is None or np.linalg.norm(long_vec) < 1e-4:
            long_vec = np.array([0, mesh.extents[1], 0])
        long_u = long_vec / np.linalg.norm(long_vec)
        axis = self._parent_axis(link)
        palmar = np.cross(axis, long_u)          # direction the finger flexes toward
        palmar /= np.linalg.norm(palmar)
        across = np.cross(long_u, palmar)
        L = max(np.ptp(mesh.vertices @ long_u), 1e-3)
        start = (mesh.vertices @ long_u).min()
        w_proj = mesh.vertices @ across
        w_mid, W = w_proj.mean(), np.ptp(w_proj)
        p_mid = (mesh.vertices @ palmar).mean()

        u = np.linspace(*span, rows)
        if C.ROW0_AT_TIP:
            u = u[::-1]
        v = np.linspace(-width / 2, width / 2, cols)
        pts = np.full((rows, cols, 3), np.nan)
        nrm = np.tile(palmar, (rows, cols, 1)).astype(float)
        origins, dirs, ij = [], [], []
        for i, ui in enumerate(u):
            for k, vk in enumerate(v):
                base = (start + ui * L) * long_u + (w_mid + vk * W) * across + p_mid * palmar
                origins.append(base - palmar * 0.05)
                dirs.append(palmar)
                ij.append((i, k))
        locs, ray_idx, tri_idx = mesh.ray.intersects_location(
            np.array(origins), np.array(dirs), multiple_hits=True)
        best = {}
        for loc, r, t in zip(locs, ray_idx, tri_idx):
            d = np.dot(loc - origins[r], palmar)
            if r not in best or d > best[r][0]:
                best[r] = (d, loc, mesh.face_normals[t])
        for r, (_, loc, n) in best.items():
            i, k = ij[r]
            pts[i, k] = loc
            nrm[i, k] = n if np.dot(n, palmar) > 0 else palmar
        # Rays that miss (the grid reaches past the curved fingertip or the link's edge) used to leave
        # the taxel without a position, so pressure there was ignored, e.g. a fingertip touch on the
        # first rows of a Nail patch. Snap those taxels to the nearest point on the link surface instead.
        miss = [r for r in range(len(origins)) if r not in best] if C.snap_missed_taxels() else []
        if miss:
            base = np.array(origins)[miss] + palmar * 0.05
            near, _, tri = trimesh.proximity.closest_point(mesh, base)
            for r, p, t in zip(miss, near, tri):
                i, k = ij[r]
                pts[i, k] = p
                n = mesh.face_normals[t]
                nrm[i, k] = n if np.dot(n, palmar) > -0.2 else palmar
        return pts, nrm

    def _palm_patch(self, rows, cols):
        mesh = self.link_mesh[REF]
        palmar = self._palm_normal()
        ys = np.linspace(*C.PALM_LENGTH_RANGE, rows)
        zs = np.linspace(*C.PALM_WIDTH_RANGE, cols)
        origins = np.array([[0, y, z] for y in ys for z in zs]) - palmar * 0.08
        dirs = np.tile(palmar, (len(origins), 1))
        locs, ray_idx, tri_idx = mesh.ray.intersects_location(origins, dirs, multiple_hits=True)
        pts = np.full((rows * cols, 3), np.nan)
        best = {}
        for loc, r, t in zip(locs, ray_idx, tri_idx):
            d = np.dot(loc - origins[r], palmar)
            if r not in best or d > best[r]:
                best[r] = d
                pts[r] = loc
        nrm = np.tile(palmar, (rows * cols, 1))
        return pts.reshape(rows, cols, 3), nrm.reshape(rows, cols, 3)

    def _palm_normal(self):
        """Palm-facing direction in hand frame: where the index finger flexes to."""
        if getattr(self, "_palm_n", None) is not None:
            return self._palm_n
        j = "index_proximal_joint"
        saved = self.urdf.cfg.copy()
        self.urdf.update_cfg({j: 0.0})
        tip0 = self.T("index_tip")[:3, 3]
        self.urdf.update_cfg({j: 0.3})
        tip1 = self.T("index_tip")[:3, 3]
        self.urdf.update_cfg(saved)
        n = tip1 - tip0
        n[1] = 0                                   # drop motion along the palm
        self._palm_n = n / np.linalg.norm(n)
        return self._palm_n

    def taxel_layout(self, group):
        """Cached taxel points/normals for a sensor group, in its link frame."""
        name = group["name"]
        if name in self._taxel_cache:
            return self._taxel_cache[name]
        finger = group["finger"].lower()
        kind = name.split()[-1].lower()
        if finger == "palm":
            res = (REF, *self._palm_patch(group["rows"], group["cols"]))
        else:
            lay = C.REGION_LAYOUT[kind]
            link = C.FINGER_LINKS[finger][lay["segment"]]
            res = (link, *self._patch(link, group["rows"], group["cols"],
                                      lay["span"], lay["width"]))
        self._taxel_cache[name] = res
        return res

    def taxels_in_hand(self, group):
        """Taxel points and normals in the hand frame at the current pose."""
        link, pts, nrm = self.taxel_layout(group)
        T = self.T(link) if link != REF else np.eye(4)
        p = pts.reshape(-1, 3) @ T[:3, :3].T + T[:3, 3]
        n = nrm.reshape(-1, 3) @ T[:3, :3].T
        return p.reshape(pts.shape), n.reshape(nrm.shape)
