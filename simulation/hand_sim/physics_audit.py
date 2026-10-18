"""Physics audit: solver settings, collision-geometry accuracy, penetration and self-intersection.

This is the acceptance test for the Phase-1 stability work (tighter timestep, solver iterations,
stiffer contacts, hand self-collision, clean convex hulls). It measures, it does not fix anything -
run it before and after a change to see whether the change helped.

Three independent questions:

1. Geometry - does the *collision* mesh match the *visual* mesh? A collision hull that bulges
   beyond the drawn surface makes the hand touch objects early and fabricates contacts that the
   real fingertips would never feel. Reported as the surface deviation in millimetres.
2. Penetration - how deep do bodies sink into each other during a grasp? Deep penetration is the
   "soft-body bleeding" that corrupts contact forces and therefore tactile values.
3. Self-intersection - do the fingers pass through each other? This is measured with
   mj_geomDistance, so it works whether or not hand self-collision is enabled.

Usage (from simulation/):
    ../venv/bin/python -m hand_sim.physics_audit --episodes 6
    ../venv/bin/python -m hand_sim.physics_audit --geometry-only
"""

import argparse
from collections import defaultdict

import mujoco
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from . import config as C
from .episode import HandoverEpisode
from .policy import ScriptedGraspPolicy
from .world import World

# Hand-vs-hand geometry pairs sampled for self-intersection (mesh-mesh distance is costly).
SELF_SCAN_EVERY = 10  # control ticks between self-intersection scans
PENETRATION_REPORT_MM = -1.0  # report frames penetrating deeper than this


def _body_geoms(model, body_id, group=None):
    out = []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != body_id:
            continue
        if group is not None and model.geom_group[g] != group:
            continue
        out.append(g)
    return out


def _mesh_paths(model, geom_id):
    """Source file path recorded for a mesh geom, if any (for offline geometry comparison)."""
    if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
        return None
    mid = model.geom_dataid[geom_id]
    off = model.mesh_pathadr[mid]
    if off < 0:
        return None
    end = model.names.index(b"\x00", off)
    return model.names[off:end].decode()


def solver_report(model: mujoco.MjModel) -> None:
    o = model.opt
    print("=== solver / options ===")
    for attr in ("timestep", "integrator", "solver", "iterations", "ls_iterations",
                 "noslip_iterations", "noslip_tolerance", "tolerance", "impratio", "cone"):
        print(f"  {attr:18s} {getattr(o, attr, 'n/a')}")
    filtered = bool(o.disableflags & int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT))
    print(f"  parent-child contacts filtered: {not filtered}")
    print(f"  hand self-collision:            {C.HAND_ALLOWS_SELF_COLLISION}")
    print(f"  maxhullvert:                    {C.MAX_HULL_VERT}")


def _sample_surface(mesh: trimesh.Trimesh) -> np.ndarray:
    """Points spread over a mesh's surface: its vertices, face centroids and edge midpoints.

    Sampling only vertices badly understates the bulge of a convex collision mesh. A convex hull
    spanning a concavity in the drawn surface has every one of its vertices *on* that surface - the
    part that sticks out is the middle of its faces. The fingers are concave by 3.9-7.5 mm, so this is
    not a corner case: measured vertex-only, thumb_intermediate reads 1.59 mm; the face samples are
    what reveal how far its hull actually reaches.
    """
    corners = mesh.vertices[mesh.faces]
    edges = np.concatenate([corners[:, [0, 1]].mean(axis=1),
                            corners[:, [1, 2]].mean(axis=1),
                            corners[:, [2, 0]].mean(axis=1)])
    return np.concatenate([mesh.vertices, corners.mean(axis=1), edges])


def _surface_deviation(collision_mesh: trimesh.Trimesh, visual_mesh: trimesh.Trimesh) -> tuple[float, float]:
    """(bulge_mm, gap_mm) between a collision mesh and the visual mesh of the same part.

    bulge_mm: how far the collision surface sticks out past the drawn surface. A positive bulge
              means the hand or object touches earlier than it should; this is the number that
              fabricates contacts at distances the real fingertips could never feel.
    gap_mm:   how much of the drawn surface has no collision geometry behind it (mean distance from
              the drawn surface to the nearest collision surface). A large gap is normal for a coarse
              convex decomposition of detailed CAD and is far less harmful than a bulge.

    Both directions are exact point-to-triangle distances (trimesh.proximity, which this venv can use
    because rtree is installed), with both surfaces sampled by _sample_surface rather than by their
    vertices alone.
    """
    import trimesh.proximity

    points = _sample_surface(collision_mesh)
    closest, distance, face_id = trimesh.proximity.closest_point(visual_mesh, points)
    signed = np.einsum("ij,ij->i", points - closest, visual_mesh.face_normals[face_id])
    outside = distance * (signed >= 0)
    bulge = float(outside.max() * 1000) if outside.size else 0.0
    gap = float(trimesh.proximity.closest_point(collision_mesh, _sample_surface(visual_mesh))[1].mean() * 1000)
    return bulge, gap


def _model_geom_mesh(model: mujoco.MjModel, geom_id: int) -> trimesh.Trimesh | None:
    """A geom's geometry as a trimesh in its body frame; None for a type this cannot represent.

    Mesh geoms contribute their vertices directly. Boxes and cylinders are converted with
    trimesh.creation - the palm is 7 boxes plus a cylinder, so without this it could only be reported
    as "not comparable", and the palm is exactly where the simulated and real tactile data disagree
    most (the real palm group fires in every session, the simulated one never does).
    """
    gtype, size = model.geom_type[geom_id], model.geom_size[geom_id]
    if gtype == mujoco.mjtGeom.mjGEOM_MESH:
        mid = model.geom_dataid[geom_id]
        v0, nv = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        f0, nf = model.mesh_faceadr[mid], model.mesh_facenum[mid]
        mesh = trimesh.Trimesh(vertices=np.array(model.mesh_vert[v0:v0 + nv]),
                               faces=np.array(model.mesh_face[f0:f0 + nf]), process=False)
    elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
        mesh = trimesh.creation.box(extents=2.0 * size[:3])  # MuJoCo size = half-extents
    elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
        mesh = trimesh.creation.cylinder(radius=size[0], height=2.0 * size[1], sections=64)
    else:
        return None
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_quat(np.roll(model.geom_quat[geom_id], -1)).as_matrix()
    transform[:3, 3] = model.geom_pos[geom_id]
    mesh.apply_transform(transform)
    return mesh


def _body_geometry(model: mujoco.MjModel) -> dict:
    """body name -> {"col": [collision meshes], "vis": [drawn meshes], "other": n_unrepresentable}.

    Read from the compiled model rather than from the mesh files, so this reports the geometry
    actually in use - the vendor collision meshes or the rebuilt decompositions from
    rebuild_hand_collision.py - and cannot drift out of sync with scene.py. Collision is group 2 for
    the hand and group 3 for the objects; the drawn surface is group 1 for both.
    """
    out: dict[str, dict] = {}
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[gid])
        if name is None:
            continue
        group = model.geom_group[gid]
        entry = out.setdefault(name, {"col": [], "vis": [], "other": 0})
        mesh = _model_geom_mesh(model, gid)
        if group == 1:
            if mesh is not None:
                entry["vis"].append(mesh)
        elif group in (2, 3):
            if mesh is None:
                entry["other"] += 1
            else:
                entry["col"].append(mesh)
    return out


def geometry_report() -> None:
    """Compare each collision geometry against the drawn surface of the same body, as loaded."""
    from .scene import build_model

    geometry = _body_geometry(build_model())

    print("=== hand: collision geometry vs drawn surface (as loaded in the model) ===")
    print("  bulge = collision surface sticks out past the drawn surface (early/wrong contacts)")
    print("  gap   = drawn surface with no collision behind it (coarse, mostly harmless)")
    print(f"  {'body':30s} {'col parts':>9s} {'col verts':>9s} {'vis verts':>9s} {'bulge mm':>9s} {'gap mm':>8s}")
    for name in sorted(n for n in geometry if n.startswith("right_")):
        entry = geometry[name]
        if not entry["col"]:
            note = f"{entry['other']} geoms of an unrepresentable type" if entry["other"] else "no collision"
            print(f"  {name:30s}   ({note})")
            continue
        collision = trimesh.util.concatenate(entry["col"])
        visual = trimesh.util.concatenate(entry["vis"])
        bulge, gap = _surface_deviation(collision, visual)
        flag = "  <-- bulges" if bulge > 2.0 else ""
        print(f"  {name:30s} {len(entry['col']):9d} {len(collision.vertices):9d} {len(visual.vertices):9d} "
              f"{bulge:9.2f} {gap:8.2f}{flag}")

    print("\n=== objects: collision parts vs visual mesh (as loaded in the model) ===")
    print(f"  {'object':12s} {'parts':>5s} {'verts/part':>12s} {'bulge mm':>9s} {'gap mm':>8s} {'>maxhullvert':>13s}")
    for obj_id in C.OBJECTS:
        entry = geometry[obj_id]
        visual = trimesh.util.concatenate(entry["vis"])
        devs = [_surface_deviation(part, visual) for part in entry["col"]]
        verts = [len(part.vertices) for part in entry["col"]]
        over = sum(1 for v in verts if v > C.MAX_HULL_VERT)
        print(f"  {obj_id:12s} {len(entry['col']):5d} {min(verts):5d}-{max(verts):<6d} "
              f"{max(d[0] for d in devs):9.2f} {np.mean([d[1] for d in devs]):8.2f} "
              f"{over:6d}/{len(verts):<6d}")


def _hand_self_pairs(model: mujoco.MjModel) -> list[tuple[int, int]]:
    """Hand collision geoms from different bodies, excluding the deliberately-excluded adjacency.

    Parent-child pairs (palm <-> each finger's proximal segment, and each finger segment <-> its
    neighbour) are skipped on purpose, but *not* because MuJoCo filters them — it does not, in this
    build, which is the whole reason `scene._exclude_adjacent_hand_bodies()` exists. Those pairs
    overlap by 11-13 mm by construction (the finger roots sit inside the palm volume), so counting
    them as "self-intersection" would report a constant several-millimetre overlap that no change to
    the physics could ever remove. What is left is the pairs that genuinely must not pass through
    each other: different fingers, and the thumb chain's non-adjacent segments.
    """
    geoms = [g for g in range(model.ngeom)
             if model.geom_contype[g] != 0 and model.body(model.geom_bodyid[g]).name.startswith("right_")]
    pairs = []
    for i, g1 in enumerate(geoms):
        b1 = model.geom_bodyid[g1]
        for g2 in geoms[i + 1:]:
            b2 = model.geom_bodyid[g2]
            if b1 == b2:
                continue  # same body: MuJoCo never collides these
            if model.body_parentid[b1] == b2 or model.body_parentid[b2] == b1:
                continue  # parent-child: filtered by default
            pairs.append((g1, g2))
    return pairs


def penetration_scan(episodes: int, seed: int = 0) -> None:
    """Run episodes and report penetration depth and hand self-intersection."""
    world = World(seed=seed)
    model, data = world.m, world.d
    rng = np.random.default_rng(seed)
    policy = ScriptedGraspPolicy()

    hand_bodies = {b for b in range(model.nbody) if model.body(b).name.startswith("right_")}
    self_pairs = _hand_self_pairs(model)

    deep = defaultdict(float)  # category -> deepest penetration (m, negative)
    frames_pen = defaultdict(int)
    ticks = 0
    worst_self = 0.0
    self_hits = 0
    results = []

    def category(b1, b2):
        # Only body 0 (the world) carries the ground plane; every other top-level body (the hand
        # root, the objects, the mocap carriers) is a real body and must not be lumped in here.
        if b1 == 0 or b2 == 0:
            return "floor"
        h1, h2 = b1 in hand_bodies, b2 in hand_bodies
        if h1 and h2:
            return "hand-hand"
        if h1 or h2:
            return "hand-object"
        return "object-object"

    for i_ep in range(episodes):
        obj_id = None
        episode = HandoverEpisode(world, policy, rng, obj_id=obj_id, randomize=True)
        f6 = np.zeros(6)

        while episode.tick():
            ticks += 1
            for c in range(data.ncon):
                con = data.contact[c]
                b1 = model.geom_bodyid[con.geom1]
                b2 = model.geom_bodyid[con.geom2]
                cat = category(b1, b2)
                if con.dist < deep[cat]:
                    deep[cat] = float(con.dist)
                if con.dist < PENETRATION_REPORT_MM / 1000:
                    frames_pen[cat] += 1
            if ticks % SELF_SCAN_EVERY == 0:
                for g1, g2 in self_pairs:
                    d = mujoco.mj_geomDistance(model, data, g1, g2, 0.02, None)
                    if d < 0:
                        self_hits += 1
                        worst_self = min(worst_self, d)
        results.append(episode.result)

    ok = sum(1 for r in results if r["success"])
    print(f"=== penetration over {episodes} episodes ({ticks} control ticks) ===")
    print(f"  {'category':14s} {'deepest':>10s} {'ticks < -1mm':>14s}")
    for cat in ("hand-object", "hand-hand", "object-object", "floor"):
        print(f"  {cat:14s} {deep[cat]*1000:9.2f}mm {frames_pen[cat]:14d}")
    print(f"\n  hand self-intersection: {self_hits} geom-pair overlaps, deepest {worst_self*1000:.2f}mm")
    print(f"  grasp success: {ok}/{len(results)} ({100*ok/len(results):.0f}%)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--geometry-only", action="store_true")
    ap.add_argument("--no-geometry", action="store_true")
    args = ap.parse_args()

    world = World(seed=args.seed)
    solver_report(world.m)
    if not args.no_geometry:
        print()
        geometry_report()
    if not args.geometry_only:
        print()
        penetration_scan(args.episodes, args.seed)


if __name__ == "__main__":
    main()
