# Plan: Maximize simulated handover case coverage

Goal: exercise the Inspire hand simulation across as many physically plausible
grasp configurations as possible, while keeping the scripted baseline able to
hold the object (so the dataset contains successful and failed grasps).

## Current randomization (already in code)
- Object: vegabar_29 / vegabar_38 (random per episode)
- Roll about long axis: full 360 deg
- Tilt in palm plane: +/-1.0 rad
- Tilt toward/away: +/-0.7 rad
- Pivot offset: x 0..1.5 cm, y +/-3 cm, z +/-2.5 cm
- Approach direction: wide hemisphere
- Grasp force: per-object (up to 3000 g)
- Servo stiffness: 50 Nm/rad

## Additional cases to add (prioritized)

### 1. Hand preshape variation
- Vary initial finger openness (not only fully open).
- Vary thumb opposition (yaw) at grasp start.
- Randomize which fingers lead the close (sequential vs simultaneous).

### 2. Object placement extremes
- Place object at palm edges (left/right of palm, near fingertips, near wrist).
- Place object at different heights relative to hand (above/below palm center).
- Include "shaft near head" vs "shaft near tail" for vegabar_38.

### 3. Orientation edge cases
- Full 360 roll (done) + tilts up to +/-90 deg where collision-free.
- Object flipped end-over-end (pitch 180).
- Object mirrored across palm midplane (swap thumb/little side) via yaw 180 when feasible.

### 4. Dynamics / contact variation
- Vary grasp force (500, 1000, 2000, 3000 g).
- Vary closing speed (slow/medium/fast).
- Vary object mass within plausible range (0.15-0.8 kg).
- Vary friction coefficients (low/medium/high).

### 5. Approach variation
- Approach from different directions (front, side, top) with collision checks.
- Vary approach speed and pause before close.
- Include "push into palm until contact" (already added) with different push depths.

### 6. Failure / edge cases
- Deliberately bad placements to generate dropped/rotated/slipped labels.
- Objects with no tactile contact (test fallback).
- Partial contacts (single finger).

### 7. Dataset balancing
- Ensure roughly equal counts per object and per outcome.
- Stratify by orientation bins (roll/tilt) so models see all.
- Record ground-truth pose for every frame (already).

## Implementation steps
1. Add config knobs for preshape variation and approach direction bins.
2. Extend `HandoverEpisode._sample_target` to support optional extreme placement flags.
3. Add a "case matrix" runner that enumerates combinations instead of pure random.
4. Update recorder to tag each session with case parameters in meta.json.
5. Re-run smoke tests; lower success threshold if needed.
6. Generate large dataset with `generate_dataset --workers N`.

## Metrics
- Success rate per object and per orientation bin.
- Coverage: number of unique (roll bin, tilt bin, position bin) combos hit.
- Target: >90% of bins hit with >=10 episodes each.
