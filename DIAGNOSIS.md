# Deploy failure diagnosis — 4 bags (2026-08-27), analysed 2026-08-28

Bags: `pdz_single_20260827_{162621,163807,165249,171512}` (config `deploy_pdz_v3_fulltilt_single`,
checkpoint `pdz_v3_fulltilt_20260826`). Plots + CSVs in this folder.

## TL;DR
The FRI drop is a **symptom**. Root cause: **the peg tip never centers on the hole (never < 3.4 mm
XY; hole clearance ~1 mm), descends onto the cooling-base TOP surface next to the hole, can't enter,
hunts sideways, DRIFTS FURTHER AWAY, then slips + hard-recontacts (0→10 N) — and that force
transient trips the ReceiveMultiplier=1 FRI.** Plus a **+5 N Z-bias** in the wrench (reads +5 N while
hovering, no contact).

## Consistent across ALL 4 bags (not one-off)
| Quantity | b162621 | b163807 | b165249 | b171512 |
|---|---|---|---|---|
| min lateral XY miss (mm) | 5.9 | 3.4 | 4.7 | 3.7 |
| XY miss at END (mm) | 12.9 | 32.7 | 12.6 | 31.4 |  ← drifts AWAY
| min height above socket-z (mm) | 13.2 | 13.9 | 17.6 | 13.1 |  ← blocked, never < 13 mm
| ever centered (XY<3mm)? | no | no | no | no |
| ever near socket-z (z<5mm)? | no | no | no | no |
| drop force cycle | 5→0→10.2 | 5→0→10.8 | 5→0→10.0 | 5→0→12.1 | ← identical
| hovering (no-contact) Fz | +5.0 | +4.9 | +4.9 | +5.1 | ← Z-BIAS

The drop always coincides with the run's peak force, which is always a 0→10 N re-contact transient
after the tool unloads — never during smooth motion. The slip is CONSISTENT, every bag.

## Finding 1 — the deploy start-delta is NON-ZERO; sim's is ~0 (THE core distribution mismatch)
Sim design (VERIFIED in code — `insertion_env_e2e_iiwa.py::_goal_delta_ee`, docstring L290):
- The peg is **physically spawned above the NOISY goal G = true_socket + injected_goal_error**.
- So the policy's obs delta (`fingertip − G`) is **≈ 0 at reset** — "the policy thinks it's at the goal."
- The **TRUE hole is up to `goal_anchor_lat_max` (±1.75 cm) away** from where the peg starts.
- `hand_init_pos=[0,0,0.0675]`, noise `[0.003,0.003,0.010]` → the ±3 mm is jitter about G (settling),
  tip ~rim+37.5 mm. The policy's job: obs says delta≈0, but VISION must reveal the true hole is offset
  and servo there. (Corrects an earlier wrong note that said "peg spawns above the TRUE hole.")

Deploy reality (from bags, `policy_obs[0:3]` = fingertip − socket_estimate at tick 0):
- b162621 |xy|=5.9, b163807 |xy|=8.6, b165249 |xy|=4.7, b171512 |xy|=5.1 mm — **NON-ZERO.**
- In sim this delta is ~0 at reset; in deploy it starts 5–9 mm. **The policy is handed a starting obs
  it never saw in training.** In sim, a nonzero delta means "estimate error to close via vision, from
  a ~0 baseline"; here we start it already nonzero from a different geometric convention, and the
  learned response drives the delta to GROW (5 → 12–33 mm) instead of shrinking.
- Height: tip starts 29–43 mm above socket-z ≈ sim rim+37.5 mm — **height is fine**, problem is LATERAL.

Root of the mismatch: in sim the peg is PLACED at the anchor G so obs-delta starts 0. In deploy the
preinsert/hover places the tip at the hover pose, but obs-delta is vs the republished anchor, and the
two do NOT coincide at t=0 (they're 5–9 mm apart). That nonzero start is out-of-distribution and the
policy's learned servo diverges from it.

## Finding 2 — the +5 N Z-bias is real and unhandled
The wrench reads **Fz ≈ +5 N with the arm HOVERING (no contact)** in all 4 bags. `ft_baseline_on_start`
did not zero it (or captured at a different pose). This config sets `ft_bias_base_xyz → 0` on the
assumption gravity comp is "now correct" — but the data says there is still a ~5 N residual.
Consequences: seat-detection and force-cap safety see a false +5 N floor; true contact = deviation
from +5 N (so the 0 N dips are the tool pulling UP, the 10 N spikes are ~5–7 N real contact).
`zero_force_obs=true` hides it from the policy but not from the safety/seat logic.

## Why it drifts AWAY (the core open question)
Fingertip−anchor XY grows 5 → 12–33 mm over the run: the policy servos the tip AWAY from its own
socket estimate. Either (a) the hole ANCHOR XY is wrong (hand-eye residual) so "toward true hole" per
vision ≠ toward anchor, or (b) the policy can't localize the real hole at this height without a
chamfer and wanders. Anchor DETECTION is self-consistent across runs (spread dx 0.7 mm, dy 1.4 mm),
so if the anchor is wrong it's a consistent CALIBRATION offset (hand-eye), not noise.

## Fix order (proposed)
1. **+5 N Z-bias**: recapture ft baseline at the true hover pose, or set `ft_bias_base_xyz` z≈−5.
   Removes a known confound from seat/safety. Cheap.
2. **Anchor XY vs true hole (hand-eye)**: verify the republished anchor lands on the real hole center
   (place tip over hole by hand, compare). If offset, fix hand-eye (see memory `d405-handeye-calib`).
   This is the seating blocker.
3. **Start centered**: ensure the peg is placed AT the anchor XY at policy start (obs XY delta ~0 like
   sim), not 4–9 mm off. Tighten preinsert/hole-align XY convergence before START.
4. FRI robustness (`ReceiveMultiplier`≥3, pendant) so contact transients don't kill runs while tuning.
