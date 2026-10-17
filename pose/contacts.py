"""Turn tactile grids into 3D contact points (hand frame)."""
from dataclasses import dataclass

import numpy as np

import config as C


@dataclass
class Contact:
    region: str
    point: np.ndarray      # pressure-weighted centroid (m)
    normal: np.ndarray     # unit vector pointing from the finger into the object
    pressure: float        # summed taxel value
    n_taxels: int


def extract_contacts(session, frames, hand):
    """Contacts per region plus every active taxel (for plotting)."""
    contacts, taxels = [], []
    for g in session.groups:
        grid = session.region_grid(frames, g)
        pts, nrm = hand.taxels_in_hand(g)
        valid = ~np.isnan(pts[..., 0])
        active = valid & (grid > C.CONTACT_THRESHOLD)
        if active.any():
            taxels.append((pts[active], grid[active]))
        if not active.any() or grid[active].sum() < C.MIN_REGION_SUM:
            continue
        w = grid[active]
        p = (pts[active] * w[:, None]).sum(0) / w.sum()
        n = (nrm[active] * w[:, None]).sum(0)
        contacts.append(Contact(g["name"], p, n / np.linalg.norm(n),
                                float(w.sum()), int(active.sum())))
    return contacts, taxels
