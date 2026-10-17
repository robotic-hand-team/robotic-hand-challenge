"""Rebuild the hand's collision geometry as clean convex decompositions of the drawn surface.

Why this exists
---------------
The vendor collision geometry is not the drawn hand. Measured against the drawn surface of the same
body (physics_audit.py --geometry-only) it bulges by up to 7.2 mm on the fingers and 18.3 mm on the
palm, so hand-object contact fires far earlier than the drawing implies, hand self-collision makes
the fingers block each other before they visually touch, and the tactile patches - which tactile.py
derives from these same group-2 geoms - sit on the inflated shape rather than on the hand.

A plain convex hull of the drawn mesh is NOT good enough: the finger links are concave by 3.9-7.5 mm
(deepest vertex inside their own hull), so a hull bridges real concavities - thumb_proximal_base would
gain 75% volume. The links need a decomposition, exactly as the objects get in assets.py.

The palm is the worst offender and a special case: it has no collision mesh at all, only 7 boxes and a
cylinder, and boxes cannot follow a curved tapered palm. There is a right_base_link.obj next to the
other collision meshes but no geom references it, and it turns out to be an exact copy of the drawn
palm mesh - so the drawn surface is already available as a decomposition input.

What gets rebuilt (targets are discovered from the model, not from the file listing)
------------------------------------------------------------------------------------
    key = the collision mesh's asset name   when the link collides as a mesh. The vendor shares one
                                            mesh across all four fingers, so 12 bodies map to 8 keys.
    key = the body name                     when the link collides as primitives (the palm).

Output
------
Inspire_hand/collision_rebuilt/<key>_part<NN>.obj plus manifest.json. Nothing is overwritten: the
vendor files stay, and scene.py swaps the geometry at load time only when the manifest is present, so
deleting the output directory restores the previous geometry exactly. Runs for different keys
accumulate, so the fingers and the palm can be tuned with separate thresholds.

Usage (from simulation/):
    ../venv/bin/python -m hand_sim.rebuild_hand_collision --measure-only         # compare, write nothing
    ../venv/bin/python -m hand_sim.rebuild_hand_collision --threshold 0.02      # all targets
    ../venv/bin/python -m hand_sim.rebuild_hand_collision --keys right_hand_root --threshold 0.05
"""

import argparse
import json
import time

import coacd
import mujoco
import numpy as np
import trimesh

from . import config as C

REBUILT_DIR = C.SIM_DIR / "Inspire_hand" / "collision_rebuilt"
MANIFEST = REBUILT_DIR / "manifest.json"


def targets() -> tuple[dict[str, trimesh.Trimesh], dict[str, trimesh.Trimesh]]:
    """({key: drawn mesh}, {key: collision geometry currently in use}), both from the compiled model.

    Reading the model rather than the file listing is what makes the before/after honest: the palm's
    file next to the other collision meshes is unused, so a file-based "before" would report the palm
    as perfect while the model actually collides with 7 boxes that stick out 18.3 mm.
    """
    from .physics_audit import _body_geometry
    from .scene import build_model

    model = build_model()
    per_body = _body_geometry(model)
    drawn: dict[str, trimesh.Trimesh] = {}
    current: dict[str, trimesh.Trimesh] = {}
    for gid in range(model.ngeom):
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[gid])
        if body is None or not body.startswith("right_") or model.geom_group[gid] != 2:
            continue
        if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
            key = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, model.geom_dataid[gid])
        else:
            key = body  # the palm collides as primitives rather than as a mesh
        if key in drawn:
            continue
        drawn[key] = trimesh.util.concatenate(per_body[body]["vis"])
        current[key] = trimesh.util.concatenate(per_body[body]["col"])
    return drawn, current


def decompose(mesh: trimesh.Trimesh, threshold: float, max_hulls: int):
    """CoACD decomposition. The mesh is already in metres and in the body frame, so - unlike
    assets.py - nothing is scaled or re-centred: these parts must stay exactly where the drawing is."""
    return coacd.run_coacd(
        coacd.Mesh(mesh.vertices, mesh.faces),
        threshold=threshold,
        max_convex_hull=max_hulls,
        preprocess_mode="auto",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threshold", type=float, default=0.02,
                    help="CoACD concavity threshold; lower = more parts, better fidelity (objects use 0.04)")
    ap.add_argument("--max-hulls", type=int, default=32)
    ap.add_argument("--keys", nargs="*", default=None,
                    help="subset of targets to rebuild (default: all). Lets the palm use its own threshold.")
    ap.add_argument("--measure-only", action="store_true",
                    help="report current vs rebuilt fidelity without writing anything")
    args = ap.parse_args()

    from .physics_audit import _surface_deviation

    drawn, current = targets()
    selected = sorted(drawn) if args.keys is None else [k for k in args.keys if k in drawn]
    if args.keys is not None and not selected:
        ap.error(f"no known targets in {args.keys}; known: {sorted(drawn)}")

    manifest = {}
    if MANIFEST.exists() and not args.measure_only:
        manifest = json.loads(MANIFEST.read_text())
    if not args.measure_only:
        REBUILT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"{'target':30s} {'parts':>5s} {'cur bulge':>10s} {'new bulge':>10s} {'cur gap':>8s} {'new gap':>8s} {'secs':>6s}")
    for key in selected:
        visual = drawn[key]
        cur_bulge, cur_gap = _surface_deviation(current[key], visual)

        t0 = time.perf_counter()
        parts = decompose(visual, args.threshold, args.max_hulls)
        elapsed = time.perf_counter() - t0
        meshes = [trimesh.Trimesh(verts, faces) for verts, faces in parts]
        merged = trimesh.util.concatenate(meshes)
        new_bulge, new_gap = _surface_deviation(merged, visual)
        counts = [len(m.vertices) for m in meshes]

        print(f"{key:30s} {len(parts):5d} {cur_bulge:9.2f}m {new_bulge:9.2f}m "
              f"{cur_gap:7.2f}m {new_gap:7.2f}m {elapsed:6.1f}  verts {min(counts)}-{max(counts)}")

        if not args.measure_only:
            for old in REBUILT_DIR.glob(f"{key}_part*.obj"):
                old.unlink()
            names = []
            for i, mesh in enumerate(meshes):
                name = f"{key}_part{i:02d}.obj"
                mesh.export(REBUILT_DIR / name)
                names.append(name)
            manifest[key] = {"parts": names, "bulge_mm": new_bulge, "gap_mm": new_gap,
                             "before_bulge_mm": cur_bulge, "before_gap_mm": cur_gap,
                             "coacd_threshold": args.threshold, "coacd_max_hulls": args.max_hulls}

    if not args.measure_only:
        MANIFEST.write_text(json.dumps(manifest, indent=2))
        worst = max((v["bulge_mm"] for v in manifest.values()), default=0.0)
        total = sum(len(v["parts"]) for v in manifest.values())
        print(f"\n{total} parts over {len(manifest)} targets -> {MANIFEST.relative_to(C.SIM_DIR.parent)}")
        print(f"worst remaining bulge: {worst:.2f} mm")


if __name__ == "__main__":
    main()
