"""Fit a known object mesh (STL) to tactile contact points.

Pose = rotation R and translation t that map object-frame points into the
hand frame: p_hand = R @ p_obj + t.
"""
from dataclasses import dataclass

import numpy as np
import trimesh
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

import config as C


class ObjectModel:
    """Object mesh centred on its bounding box with its long axis along z."""

    def __init__(self, stl_path, name=None):
        mesh = trimesh.load(stl_path, force="mesh")
        mesh.apply_scale(C.OBJECT_UNITS_TO_M)
        if len(mesh.faces) > C.OBJECT_MAX_FACES:
            try:
                mesh = mesh.simplify_quadric_decimation(face_count=C.OBJECT_MAX_FACES)
            except Exception:
                pass                              # decimation optional
        # object frame: principal axes, origin at bounding-box centre
        to_axes = mesh.principal_inertia_transform
        mesh.apply_transform(to_axes)
        ext = mesh.extents
        order = np.argsort(ext)                   # make z the longest axis
        perm = np.eye(4)
        perm[:3, :3] = np.eye(3)[[order[0], order[1], order[2]]]
        if np.linalg.det(perm[:3, :3]) < 0:
            perm[0] *= -1
        mesh.apply_transform(perm)
        centre = mesh.bounds.mean(0)
        mesh.apply_translation(-centre)
        shift = np.eye(4)
        shift[:3, 3] = -centre
        self.cad_to_obj = shift @ perm @ to_axes  # original CAD (scaled) -> object frame
        self.mesh = mesh
        self.name = name or str(stl_path)
        self.radius = 0.5 * min(ext[order[0]], ext[order[1]])
        self.length = ext[order[2]]
        pts, face = trimesh.sample.sample_surface_even(mesh, 25000, seed=0)
        self.surf_pts = pts
        self.surf_nrm = mesh.face_normals[face]
        self.tree = cKDTree(pts)

    def signed_distance(self, p_obj):
        """Approximate signed distance (outside > 0) and outward normals."""
        _, idx = self.tree.query(p_obj)
        q, n = self.surf_pts[idx], self.surf_nrm[idx]
        return np.einsum("ij,ij->i", p_obj - q, n), n


@dataclass
class FitResult:
    T: np.ndarray            # 4x4 object frame -> hand frame
    rms_mm: float
    normal_err_deg: float
    max_penetration_mm: float
    n_contacts: int
    ambiguous: bool
    alternatives: int
    collides: bool = False

    @property
    def axis(self):
        return self.T[:3, 2]

    @property
    def ok(self):
        return self.rms_mm < C.GOOD_FIT_RMS_MM and not self.ambiguous and not self.collides


def _to_T(x):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(x[:3]).as_matrix()
    T[:3, 3] = x[3:]
    return T


# Main directions in the hand frame (hand_base_link)
AXIS_HINTS = {
    "across_palm": np.array([0.0, 0.0, 1.0]),   # index side <-> little-finger side
    "along_palm": np.array([0.0, 1.0, 0.0]),    # wrist <-> fingertips (parallel to fingers)
    "out_of_palm": np.array([1.0, 0.0, 0.0]),   # pointing away from the palm
}


def axis_hint_vector(hint):
    if hint is None:
        return None
    v = AXIS_HINTS[hint] if isinstance(hint, str) else np.asarray(hint, float)
    return v / np.linalg.norm(v)


def fit_pose(obj: ObjectModel, contacts, hand_points=None, n_random=C.N_ORIENTATION_STARTS,
             seed=0, axis_hint=C.AXIS_HINT, check_points=None):
    """Multi-start least-squares fit. Returns FitResult or None."""
    if len(contacts) < 2:
        return None
    P = np.array([c.point for c in contacts])
    Nf = np.array([c.normal for c in contacts])
    w = np.sqrt(np.array([c.pressure for c in contacts]))
    w = np.maximum(w / w.max(), C.CONTACT_WEIGHT_FLOOR)
    centroid = (P * w[:, None]).sum(0) / w.sum()
    inward = Nf.mean(0)
    inward = inward / (np.linalg.norm(inward) + 1e-9)
    H = hand_points if hand_points is not None else np.zeros((0, 3))

    f_rob = 0.003      # robust scale for contact terms (m): outlying contacts count less

    def robust(r):
        """Soft-L1 applied by hand, so that hand-overlap terms can stay fully quadratic."""
        z = (r / f_rob) ** 2
        return np.sign(r) * f_rob * np.sqrt(2 * (np.sqrt(1 + z) - 1))

    def make_residuals(Hset):
        def residuals(x):
            R = Rotation.from_rotvec(x[:3]).as_matrix()
            t = x[3:]
            d, n_obj = obj.signed_distance((P - t) @ R)
            n_hand = n_obj @ R.T
            r_dist = robust(w * d)
            r_norm = robust(C.NORMAL_WEIGHT * w * (1 + np.einsum("ij,ij->i", n_hand, Nf)))
            if len(Hset):
                dh, _ = obj.signed_distance((Hset - t) @ R)
                r_pen = C.PENETRATION_WEIGHT * np.minimum(dh + 0.0015, 0)   # not softened
            else:
                r_pen = np.zeros(0)
            r_prior = C.PRIOR_WEIGHT * (t - t_prior)
            r_hint = (np.array([C.AXIS_HINT_WEIGHT * (1 - abs(R[:, 2] @ hint))])
                      if hint is not None else np.zeros(0))
            return np.concatenate([r_dist, r_norm, r_pen, r_prior, r_hint])
        return residuals

    hint = axis_hint_vector(axis_hint)

    # starting guesses: object centre one radius "inside" the grasp. When contacts
    # push from opposite sides (pinch: fingertips vs palm) the mean normal is weak and
    # the centre lies between them, so don't offset it.
    spread = min(1.0, float(np.linalg.norm(Nf.mean(0))))
    t_prior = centroid + inward * obj.radius * spread
    centres = [centroid + inward * obj.radius, centroid]   # wrap grasp, pinch grasp
    t0 = centres[0]
    starts = []
    for R in Rotation.random(n_random, random_state=seed):
        starts.append(np.r_[R.as_rotvec(), t0])
    # structured starts: long axis along the hint, or along each main hand
    # direction when there is no hint (so no grasp type is favoured)
    dirs = [hint] if hint is not None else list(AXIS_HINTS.values())
    for d in dirs:
        for flip in (1, -1):
            base = Rotation.align_vectors([flip * d], [[0, 0, 1]])[0]
            for spin in (0, np.pi):
                R = base * Rotation.from_rotvec([0, 0, spin])
                for c0 in centres:
                    for s in (-0.4, 0, 0.4):
                        starts.append(np.r_[R.as_rotvec(), c0 + flip * d * s * obj.length])

    # Stage 1: coarse search from every start (sparse hand points, fast)
    coarse = make_residuals(H)
    sols = []
    for x0 in starts:
        res = least_squares(coarse, x0, max_nfev=150)
        sols.append((res.cost, res.x))
    sols.sort(key=lambda s: s[0])

    # Stage 2: refine the best distinct candidates against the dense hand surface, so the
    # object cannot sink into the fingers between sparse sample points.
    Hc = check_points if check_points is not None else H
    fine = make_residuals(Hc)
    picked = []
    for c, x in sols:
        Tx = _to_T(x)
        if all(np.linalg.norm(Tx[:3, 3] - _to_T(y)[:3, 3]) > 0.004 or
               abs(Tx[:3, 2] @ _to_T(y)[:3, 2]) < 0.985 for y in picked):
            picked.append(x)
        if len(picked) == C.N_REFINE:
            break
    sols = []
    for x0 in picked:
        res = least_squares(fine, x0, max_nfev=100)
        sols.append((res.cost, res.x))
    sols.sort(key=lambda s: s[0])

    # Physics check: the object may not pass through the hand. Measured on the dense
    # set of hand surface points; candidates deeper than MAX_PENETRATION_MM are rejected.

    def penetration_mm(x):
        if not len(Hc):
            return 0.0
        Ti = _to_T(x)
        dh, _ = obj.signed_distance((Hc - Ti[:3, 3]) @ Ti[:3, :3])
        return max(0.0, -float(dh.min()) * 1000)

    feasible = []
    for c, x in sols:
        p = penetration_mm(x)
        if p <= C.MAX_PENETRATION_MM:
            feasible.append((c, x, p))
    collides = not feasible
    if collides:                         # nothing physical: take the least-colliding one
        scored = sorted(((penetration_mm(x), c, x) for c, x in sols[:10]), key=lambda s: s[0])
        feasible = [(scored[0][1], scored[0][2], scored[0][0])]
    best_cost, xb, pen = feasible[0]
    sols = [(c, x) for c, x, _ in feasible]
    T = _to_T(xb)

    # quality numbers
    R, t = T[:3, :3], T[:3, 3]
    d, n_obj = obj.signed_distance((P - t) @ R)
    ang = np.degrees(np.arccos(np.clip(-np.einsum("ij,ij->i", n_obj @ R.T, Nf), -1, 1)))

    # ambiguity: other near-optimal solutions with a clearly different pose
    alts = 0
    for c, x in sols[1:]:
        if c > best_cost * 1.15 + 1e-9:
            break
        Ti = _to_T(x)
        dpos = np.linalg.norm(Ti[:3, 3] - t) * 1000
        dax = np.degrees(np.arccos(np.clip(abs(Ti[:3, 2] @ T[:3, 2]), -1, 1)))
        if dpos > 5 or dax > 10:
            alts += 1
    return FitResult(T, float(np.sqrt(np.mean(d ** 2)) * 1000), float(ang.mean()), pen,
                     len(contacts), alts > 0, alts, collides)
