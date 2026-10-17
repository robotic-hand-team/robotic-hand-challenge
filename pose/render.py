"""3D output: a static PNG and an interactive HTML view of hand + object."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Appearance: opacity 1.0 = solid, lower = see-through (shows contacts hidden behind fingers)
HAND_COLOR = "#9aa0a8"
HAND_OPACITY = 1.0
OBJECT_COLOR = "#1d9e75"
OBJECT_OPACITY = 1.0


_LIGHT_CACHE = {}


def _light(mesh, decimate=2500):
    """Simplify for drawing, but keep the original if simplification damages the shape."""
    if len(mesh.faces) <= decimate:
        return mesh
    key = (len(mesh.faces), round(float(mesh.area), 9), decimate)
    if key in _LIGHT_CACHE:
        ok = _LIGHT_CACHE[key]
    else:
        try:
            d = mesh.simplify_quadric_decimation(face_count=decimate)
            ok = d.area > 0.9 * mesh.area          # lost surface = holes in the drawing
        except Exception:
            ok = False
        _LIGHT_CACHE[key] = ok
    if not ok:
        return mesh
    return mesh.simplify_quadric_decimation(face_count=decimate)


def _status_lines(info):
    r = info.get("fit")
    head = f"{info['session']} | object: {info['object']} | hint: {info.get('hint') or 'none'}"
    if r is None:
        return head, "No pose: fewer than 2 contact regions"
    ax = r.axis
    flag = ("COLLIDES WITH HAND" if r.collides else "OK" if r.ok
            else "AMBIGUOUS" if r.ambiguous else "POOR FIT")
    return head, (f"{flag} | contacts {r.n_contacts} | fit RMS {r.rms_mm:.1f} mm | "
                  f"normal err {r.normal_err_deg:.0f} deg | overlap {r.max_penetration_mm:.1f} mm | "
                  f"axis [{ax[0]:+.2f} {ax[1]:+.2f} {ax[2]:+.2f}]")


LIGHT_DIR = np.array([0.4, -0.5, 0.75]) / np.linalg.norm([0.4, -0.5, 0.75])


def _faces(m, color, alpha):
    """Polygons plus per-face RGBA with simple two-sided diffuse lighting."""
    polys = m.vertices[m.faces] * 1000
    lam = np.abs(m.face_normals @ LIGHT_DIR)
    shade = 0.45 + 0.55 * lam
    rgb = np.array(to_rgb(color))[None, :] * shade[:, None]
    rgba = np.column_stack([np.clip(rgb, 0, 1), np.full(len(rgb), alpha)])
    return polys, rgba


def save_png(path, hand_meshes, obj_mesh, contacts, taxels, info):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    # One collection for hand + object, so matplotlib depth-sorts them together
    parts = [_faces(_light(m, 6000 if link == "hand_base_link" else 2500), HAND_COLOR,
                    HAND_OPACITY) for link, m in hand_meshes.items()]
    if obj_mesh is not None:
        parts.append(_faces(_light(obj_mesh, 30000), OBJECT_COLOR, OBJECT_OPACITY))
    polys = np.concatenate([p for p, _ in parts])
    colors = np.concatenate([c for _, c in parts])
    ax.add_collection3d(Poly3DCollection(polys, facecolors=colors, edgecolors=colors,
                                         linewidths=0.1))
    for pts, vals in taxels:
        ax.scatter(*(pts * 1000).T, c=vals, cmap="inferno", s=6, vmin=0, vmax=1500)
    for c in contacts:
        p = c.point * 1000
        ax.scatter(*p, color="#d85a30", s=60, edgecolor="k")
        ax.quiver(*p, *(c.normal * 12), color="#d85a30")
    allp = np.vstack([m.vertices for m in hand_meshes.values()]) * 1000
    lo, hi = allp.min(0), allp.max(0)
    mid, half = (lo + hi) / 2, (hi - lo).max() / 2
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(mid[2] - half, mid[2] + half)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)"); ax.set_zlabel("z (mm)")
    ax.view_init(elev=20, azim=-150)
    t1, t2 = _status_lines(info)
    ax.set_title(f"{t1}\n{t2}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _mesh3d(m, color, opacity, name):
    v = m.vertices * 1000
    f = m.faces
    return go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2],
                     color=color, opacity=opacity, name=name, flatshading=False,
                     lighting=dict(ambient=0.45, diffuse=0.8, specular=0.25, roughness=0.6),
                     lightposition=dict(x=100, y=200, z=300),
                     hoverinfo="name", showlegend=False)


def save_html(path, hand_meshes, obj_mesh, contacts, taxels, info):
    data = [_mesh3d(_light(m, 8000), HAND_COLOR, HAND_OPACITY, link)
            for link, m in hand_meshes.items()]
    if obj_mesh is not None:
        data.append(_mesh3d(_light(obj_mesh, 60000), OBJECT_COLOR, OBJECT_OPACITY, info["object"]))
    if taxels:
        pts = np.vstack([p for p, _ in taxels]) * 1000
        vals = np.concatenate([v for _, v in taxels])
        data.append(go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers",
                                 marker=dict(size=2.5, color=vals, colorscale="Inferno",
                                             cmin=0, cmax=1500,
                                             colorbar=dict(title="taxel")),
                                 name="active taxels", hoverinfo="text",
                                 text=[f"{v:.0f}" for v in vals]))
    for c in contacts:
        p, e = c.point * 1000, (c.point + c.normal * 0.012) * 1000
        data.append(go.Scatter3d(x=[p[0], e[0]], y=[p[1], e[1]], z=[p[2], e[2]],
                                 mode="lines+markers", line=dict(color="#d85a30", width=5),
                                 marker=dict(size=[6, 0], color="#d85a30"),
                                 name=c.region, hoverinfo="text",
                                 text=f"{c.region}: {c.pressure:.0f} ({c.n_taxels} taxels)"))
    t1, t2 = _status_lines(info)
    fig = go.Figure(data)
    fig.update_layout(title=dict(text=f"{t1}<br><sup>{t2}</sup>"),
                      scene=dict(aspectmode="data", xaxis_title="x (mm)",
                                 yaxis_title="y (mm)", zaxis_title="z (mm)"),
                      margin=dict(l=0, r=0, t=60, b=0), showlegend=False)
    fig.write_html(path, include_plotlyjs=True, full_html=True)
