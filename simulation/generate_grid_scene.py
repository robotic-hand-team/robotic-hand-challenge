"""Generate a grid scene with N x N hands (static, no control).

Usage (from simulation/):
    ../venv/bin/python generate_grid_scene.py --n 10 --out scene_100.xml
View:
    ../venv/bin/mjpython -m mujoco.viewer --xml scene_100.xml
"""
import argparse
import mujoco

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--spacing", type=float, default=0.35)
    ap.add_argument("--out", default="scene_grid.xml")
    args = ap.parse_args()

    base = mujoco.MjSpec.from_file("Inspire_hand/xml/inspire_hand_right.xml")
    root = mujoco.MjSpec()
    root.copy_during_attach = True
    for i in range(args.n):
        for j in range(args.n):
            frame = root.worldbody.add_frame(name=f"f_{i}_{j}",
                                            pos=[i * args.spacing, j * args.spacing, 0])
            root.attach(base, f"h{i}_{j}_", frame=frame)
    xml = root.to_xml()
    with open(args.out, "w") as f:
        f.write(xml)
    m = root.compile()
    print(f"Generated {args.n*args.n} hands -> {args.out} (bodies={m.nbody}, actuators={m.nu})")

if __name__ == "__main__":
    main()
