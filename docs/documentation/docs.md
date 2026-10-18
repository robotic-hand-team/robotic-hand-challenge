# `docs/`: reference material

This folder holds the project documentation, the challenge brief, the hardware manuals and planning notes. There is no code here.

## Contents

| File | What it is |
|---|---|
| `documentation/` | **One documentation file per repo folder:** [backend](backend.md) · [classifier](classifier.md) · [demo](demo.md) · [docs](docs.md) · [final_model](final_model.md) · [frontend](frontend.md) · [hardware](hardware.md) · [matcher](matcher.md) · [models](models.md) · [pose](pose.md) · [simulation](simulation.md) |
| `sim_case_coverage_plan.md` | Plan for widening the simulated grasp scenarios: hand preshape, placement extremes, orientation edge cases, force/speed/mass/friction variation |
| `internal/challenge_description.md` | The Black Forest Hackathon 2026 challenge: estimate the position and orientation of a grasped object and detect slip, from touch only |
| `internal/hand_manual.pdf` | **Inspire RH56DFTP user manual.** The source of truth for the hand. Register table on p.16–17, tactile sensor layout on p.26–27 |
| `internal/INSPIRE-ROBOTS-WRIST-MODULE-OF-THE-DEXTEROUS-HAND-INSTRUCTIONS.pdf` | Manual of the wrist module |
| `internal/wrist_module_relevant_information.md` | Summary of the wrist registers (1020–1042, 100 units per degree). Not used by the code yet |

## Result figures

The benchmark figures are in the **repo root**, not in this folder:

| Image | Shows |
|---|---|
| `grasp_positions_simple.png` | Overview of grasp positions |
| `pose_preview.png` | Preview of an estimated pose |
| `position_and_orientation.png` | Top-down tilt directions and tilt histogram (written by `classifier/visualize_orientation.py`) |
| `pred_vs_gt.png` | Predicted vs ground-truth object position |
