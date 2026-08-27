# Pre-deploy checklist — pdz_v3_fulltilt (2026-08-27)

Deploying the `pdz_v3_fulltilt_20260826` run (ram-fix + corrected gravity comp + 30mm geometry,
full 25° tilt curriculum). Config already prepped: `config/deploy_pdz_v3_fulltilt.yaml`.

Work top-to-bottom. **Do NOT skip the no-contact force check (step 4)** — it's the one that catches
the gravity-comp mismatch that burned the last deploy.

---

## 0. Confirm training finished cleanly (train box)
- [ ] Run is done / at target: `grep "MAX EPOCHS\|epoch: 2000" /tmp/pdz_v3_fulltilt_20260826.log`
- [ ] Final checkpoint exists: `ls -la logs/rl_games/Forge/pdz_v3_fulltilt_20260826/nn/Forge.pth`
- [ ] No NaN/obs-guard in the log:
      `grep -iE "obs-guard|traceback|[^a-z]nan[^a-z]" /tmp/pdz_v3_fulltilt_20260826.log | grep -iv futurewarning`
- [ ] Sanity numbers (want success high @ FULL tilt, force low):
      `~/miniconda3/envs/isaaclab51/bin/python scripts/plot_e2e_progress.py --full --runs pdz_v3_fulltilt_20260826 --out diagnostics/plots/predeploy_check.png`
      → success last-50 should be strong; contact_force mean well under the 12N penalty threshold.
- [ ] (optional but recommended) render 5 full-tilt clips, eyeball the descent/bounce:
      see the render command in the run memory note.

## 1. Build the deploy package (deploy box)
- [ ] `cd ~/Masterthesis-rl-deploy`
- [ ] `colcon build --packages-select rl_deploy_inference`   ← REQUIRED: installs the new yaml
- [ ] `source install/setup.bash`
- [ ] Confirm the yaml shipped:
      `ls install/rl_deploy_inference/share/rl_deploy_inference/config/deploy_pdz_v3_fulltilt.yaml`

## 2. Robot + perception up
- [ ] Arm driver / FRI up, robot in the pre-insert home, E-stop in reach.
- [ ] RealSense streaming: `ros2 topic hz /realsense_1/camera/color/image_raw` (expect ~15-30 Hz)
- [ ] Hole aligner ready to publish the corrected anchor (see hole_align / --republish-anchor).

## 3. Launch inference (GATED — no motion yet)
- [ ] `ros2 launch rl_deploy_inference deploy_inference_pdz.launch.py \`
      `    params_file:=$(ros2 pkg prefix rl_deploy_inference)/share/rl_deploy_inference/config/deploy_pdz_v3_fulltilt.yaml`
- [ ] Node comes up, subscribes, streams the held pose. `policy_active_on_start:=false` so it will NOT move.

## 4. ⚠️ NO-CONTACT FORCE CHECK (the critical gate — do BEFORE any motion)
The new run trained on PURE contact force (corrected gravity comp) and `ft_bias_base_xyz=0`. So at a
free-space hover the policy force channels MUST read ~0.
- [ ] Hover the tool in free space (no screw touching anything), tool pointing down.
- [ ] Read the debug obs (publish_debug_obs:=true is set):
      `ros2 topic echo /rl_deploy/debug_obs`   → force is indices **[6:9]** of the 15-D vector.
- [ ] **PASS = all three force values ≈ 0** (a few tenths of N is fine).
- [ ] **FAIL = a large DC offset (esp. a negative TCP-z ~5N).** If so: STOP. The real gravity comp
      isn't matching sim. Do NOT set ft_bias back to 5.3 (that was the OLD broken-sim workaround).
      Recheck: ft_baseline_on_start captured at a clean no-contact pose? force_source? ft_sign?
- [ ] Also confirm the force SIGN: press the tip gently by hand → TCP-z force should go POSITIVE.
      If negative, flip `ft_sign` (currently -1.0).

## 5. Geometry spot-check (quick, avoids a 12mm-class error)
- [ ] `screw_tip_offset_tcp_xyz = [0,0,0.030]` matches the real tip-to-TCP you measured (~30mm). ✓ nominal.
- [ ] `socket_depth_m = 0.0175` matches the real socket depth.
- [ ] `tip_link` (lbr_two_gripper_tcp) resolves in the real robot_description (FK gives a sane TCP pose).
- [ ] Anchor: the hole-aligner publishes the detected OPENING; the node subtracts socket_depth → bottom.
      One echo of `/rl_deploy/corrected_socket` should sit at the real hole opening.

## 6. First trial (armed)
- [ ] Screw grasped, arm at pre-insert hover (~tip 3-4cm above the hole, near-vertical if possible —
      the policy is strongest near the tilt it saw; extreme tilt is the untested edge).
- [ ] Hand on E-stop.
- [ ] Arm motion: `ros2 service call /rl_deploy/start_policy std_srvs/srv/Trigger {}`
- [ ] WATCH: does it descend gently and seat, or ram? `ft_force_cap_n=25` will abort on a hard jam.
      Expected (from sim): mean contact ~5N, gentle approach. A hard ram = investigate before retrying.
- [ ] Log the outcome + |F|max + whether it seated, exactly like the last 3-trial table.

## If it rams anyway (contingency — matches the sim watch-list)
The sim levers to soften contact, in order of preference:
1. Soften arm PD: `e2e_arm_stiffness_scale < 1.0` (retrain) — makes contact compliant not rigid.
2. Halve action scale again: 0.005 → 0.0025 (retrain) — gentler max command.
Neither is a same-day fix; note it and fall back to reviewing the descent dynamics.

---
Prepped 2026-08-26. Config diff vs ep1500 = checkpoint paths + ft_bias 5.3→0 + action_scale 0.01→0.005.
No deploy-code changes were needed (force pipeline was already pure-contact).
