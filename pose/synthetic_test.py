"""Check the fitter: known pose -> fake contacts (+noise) -> fit -> compare."""
import numpy as np
from scipy.spatial.transform import Rotation

import config as C
from contacts import Contact
from pose_fit import ObjectModel, fit_pose


def run(obj, n_trials=5, n_contacts=7, noise_mm=0.5, seed=3):
    rng = np.random.default_rng(seed)
    for k in range(n_trials):
        # true pose: long axis roughly across the palm, random tilt up to 30 deg
        tilt = Rotation.from_rotvec(rng.normal(size=3) * np.radians(15))
        R = tilt.as_matrix()
        t = np.array([-0.03, -0.11, 0.0]) + rng.normal(size=3) * 0.005
        # contacts: surface points from one side of the object, middle third of its length
        idx = rng.choice(len(obj.surf_pts), 4000, replace=False)
        p, n = obj.surf_pts[idx], obj.surf_nrm[idx]
        keep = (np.abs(p[:, 2]) < obj.length / 3) & (np.abs(n[:, 2]) < 0.3)
        p, n = p[keep][:n_contacts], n[keep][:n_contacts]
        P = p @ R.T + t + rng.normal(size=p.shape) * noise_mm / 1000
        N = -(n @ R.T)
        contacts = [Contact(f"c{i}", P[i], N[i], 1000.0, 10) for i in range(len(P))]
        fit = fit_pose(obj, contacts, n_random=16)
        dpos = np.linalg.norm(fit.T[:3, 3] - t) * 1000
        e = fit.T[:3, 3] - t
        along = abs(e @ R[:, 2]) * 1000
        lateral = np.linalg.norm(e - (e @ R[:, 2]) * R[:, 2]) * 1000
        dax = np.degrees(np.arccos(abs(fit.T[:3, 2] @ R[:, 2])))
        print(f"trial {k}: position error {dpos:5.1f} mm | axis error {dax:4.1f} deg | "
              f"along {along:4.1f} / lateral {lateral:4.1f} mm | fit RMS {fit.rms_mm:.2f} mm | ambiguous={fit.ambiguous}")


if __name__ == "__main__":
    run(ObjectModel(C.OBJECTS["standin"], "standin"))
