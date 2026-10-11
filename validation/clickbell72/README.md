# ClickBell learned-mask baseline: 72/100

Frozen 2026-10-11 13:36 CST. Collected 60 scenes / 1,800 paired RGB-label images; split by scene into 54 scenes / 1,620 training images and 6 scenes / 180 validation images. SegFormer-B0 trained for 3,000 steps. Simulator object IDs supplied training labels only. Inference predicts background/robot/bell from RGB; robot and bell are rendered red(255,0,0), background gray(128,128,128), preserving the older team's 50K policy input. No additional SmolVLA training was performed for this version.

Evaluation: real ClickBell, current bundled random configuration, 361 steps, official expert-planning feasibility filter, independent evaluation seeds starting9200000 (training starts9100000), no simulator mask/depth sensor output, 50-step predicted and executed chunks. 100 distinct completed feasible scenes,72success/28timeout. All successes finished by68steps. The accompanying evaluation_complete.json records actual seeds, task-config/model hashes and diagnostic press depth. It does not contain a complete source-code hash inventory. Ground-truth diagnostics do not enter the deployed policy.

Robot/bell held-out IoU:0.95120/0.93399. These are segmentation metrics, not task success rates. This result is not the historical oracle-mask71% result and is not a hidden official evaluation score. The cloud environment's NoMaterial robot-asset compatibility choice is described in the parent README-submit.md; clean standard-environment performance remains unverified.

Portable integration changes only imports and the ClickBell task route; original nine task records remain byte-value equivalent. Other tasks and shared runtime keep the old bundle versions. Original official35% ClickBell files remain available as fallback but are no longer selected.

Portable entry smoke: 2026-10-11 14:02 CST, the actual `eval.sh` / `run_eval.py` / unified four-return policy entry ran two complete episodes with intervening reset at the same fixed feasible seed9200003. Both succeeded at48actions, workers closed normally, and offline local-model loading was enabled. See `unified-smoke.json` and `unified-smoke.log`. This is one repeated scene for deployment/reset verification, not two new independent performance samples. The cloud stage used separate copied runtime source with read-only hardlinked model/common assets to avoid duplicating disk usage; it was not a from-empty-machine installation.

Local portable-package tests:5passed. Existing bundle tests:38passed/20skipped on Windows without torch/cv2; skips remain explicit. Other nine task records and code paths were unchanged; their historical checks were not rerun as a new ten-task performance experiment.

These checks do not imply a public upload or formal update has completed.
