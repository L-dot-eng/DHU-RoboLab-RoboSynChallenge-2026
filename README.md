# DHU RoboLab — RoboSynChallenge ten-task submission

## Final selected release: ClickBell84 / drawer65 / basket54 / pipette49

Download these TWO files from [Hugging Face files](https://huggingface.co/Lhaguhgd/DHU-RoboLab-RoboSynChallenge-10Tasks-2026/tree/main):

1. `RoboSyn十任务.zip` — complete original runtime,8,683,517,915bytes; SHA256 `7e02086fcc79e2c79494ad7d1a4919da36820ae74f2a8f1fc5be919c88c9cc82`.
2. `ClickBell_84_RGBMask_Recovery_增量.zip` — final ClickBell84 overlay,741,364,011bytes; SHA256 `af6a7818c280b32dd18e2643de3ed3ce5c373b5d96760cfa9fb989021afcbbf5`.

Extract the original ZIP first, then the84overlay into the SAME parent directory, merging/overwriting the same `RoboSyn十任务_抽屉65_提篮54_移液器49_SmolVLA/` folder. Do not nest another copy of this folder. No intermediate72overlay is needed. The overlay is NOT a standalone runtime. The original ZIP alone still selects official ClickBell35.

`ClickBell_72_RGBMask_增量.zip` is an OPTIONAL alternative fallback, not an additional installation step. To use it, start from a fresh extraction of the original ZIP and apply ONLY72, not84. Its size741,270,120bytes and SHA256 `ddc2b85f5a50c11643dc0fd59654800f71fda4f33232d2f64472bcfbc0317643` identify the frozen pure-policy fallback. Do not mix the two overlays in an existing modified directory.

Enter the merged runtime. In the official Linux/NVIDIA GPU, Python3.11 / EmbodiChain0.2.4 environment, run:

```bash
python verify_bundle.py
bash setup_policy.sh
bash eval.sh click_bell random 0 --max_episodes 100 --seed 0
```

Replace `click_bell` with any task from [tasks.json](https://github.com/L-dot-eng/DHU-RoboLab-RoboSynChallenge-2026/blob/main/tasks.json). The positional0 is the GPU index. Preserve all normalizers, shared VLM/tokenizer, learned predictors, vendored dependencies, assets and license notices. [README-submit.md](https://github.com/L-dot-eng/DHU-RoboLab-RoboSynChallenge-2026/blob/main/README-submit.md) lists every task's checkpoint and run command. HF archives contain complete sources/weights; GitHub is a reviewable core-source mirror, not a standalone runtime.

## Method and measured evidence

Four team-selected policies: ClickBell84, drawer65, basket54, pipette49. The other SIX tasks, including SampleLoading, retain official SmolVLA. Scores are separate public/historical results, not a unified ten-task benchmark or hidden-test score.

ClickBell uses the team's existing50K red/gray SmolVLA weights plus an RGB SegFormer-B0 predictor:60scenes/1800paired images,54scenes/1620training and6scenes/180validation,3000predictor steps. Simulator labels supervise training only. Inference takes RGB, task instruction and robot proprioception; it never consumes simulator instance masks or bell poses. This is neither the oracle-mask71% version nor the teammate's handwritten CAP submission.

The pure-policy fallback scored72/100. The selected variant preserves its first100actions and50-predicted/50-executed queue, then permits at most five50-action local retries using RGB-predicted visibility, robot kinematics and measured action history. A lowest-TCP heuristic, not exact RGB object localization, defines the local anchor. Trials use center and±4mm X/Y offsets, with+6mm approach and−3mm press relative to the anchor, bounded/verified IK and smooth joint commands. Model weights did not change; the shared `checkpoints/click_bell_72pct_ours/` folder name identifies the unchanged base weights, while `tasks.json` selects the84recovery controller.

Independent100feasible public scenes yield84successes/16failures under official timeout-first scoring:77early successes and7during recovery. The immutable raw log has85task-predicate successes; one additional event at step361 coincides with timeout and is excluded by the official evaluator's rule. Both raw and corrected counts are disclosed. The12-point difference from72 is NOT a strict paired causal gain: initial rendered inputs differed in nominally paired development tests. The100-run experiment seeds worker noise per scene; portable deployment retains the existing no-argument worker reset, so exact per-seed reproduction is not claimed.

The actual portable unified entry passed six complete, reset-separated development episodes, including two natural recovery activations (one success at127steps, one normal361-step timeout), finite14-value actions and all reset-state assertions. These six have no success/timeout collision. This targeted smoke is interface coverage, not another unbiased performance estimate. Other nine routes/weights are unchanged. Current cloud checks used NoMaterial robot visual assets and an existing installation; a clean standard-asset environment and hidden perturbations remain unverified. Full evidence is in the HF archive's `validation/clickbell84/` and its [source mirror](https://github.com/L-dot-eng/DHU-RoboLab-RoboSynChallenge-2026/tree/main/validation/clickbell84).

Preserve [THIRD_PARTY_NOTICES.md](https://github.com/L-dot-eng/DHU-RoboLab-RoboSynChallenge-2026/blob/main/THIRD_PARTY_NOTICES.md). NVIDIA SegFormer's research/evaluation non-commercial restriction and other upstream model terms are not overridden by a top-level code license. No training trajectories, credentials, SSH keys or conversation records are included.
