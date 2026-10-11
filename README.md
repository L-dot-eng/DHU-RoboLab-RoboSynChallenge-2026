# DHU RoboLab — RoboSynChallenge ten-task submission

## Final v4: ClickBell84 / drawer65 / basket54 / pipette49 / ItemsHandover33.3

This v4 replaces only ItemsHandover in the prior ClickBell84-v3 submission. The other nine routes, controllers and weights are unchanged. Five team-selected policies and five attributed official SmolVLA policies cover all ten tasks, including official RGB SampleLoading.

Download these THREE files from [Hugging Face files](https://huggingface.co/Lhaguhgd/DHU-RoboLab-RoboSynChallenge-10Tasks-2026/tree/main) and extract in this order:

1. `RoboSyn十任务.zip`: 8,683,517,915 bytes; SHA256 `7e02086fcc79e2c79494ad7d1a4919da36820ae74f2a8f1fc5be919c88c9cc82`.
2. `ClickBell_84_RGBMask_Recovery_增量.zip`: 741,364,011 bytes; SHA256 `af6a7818c280b32dd18e2643de3ed3ce5c373b5d96760cfa9fb989021afcbbf5`.
3. `ItemsHandover_33.3pct_PredMask_增量.zip`: 868,253,284 bytes; SHA256 `f54410eed3858f8f08484748730a68b8c65f43a6e3c135cde63843fefbcdc8f0`.

Merge/overwrite the same `RoboSyn十任务_抽屉65_提篮54_移液器49_SmolVLA/` folder under one parent directory. Do not nest copies. Do NOT apply the optional ClickBell72 fallback as another step. Neither overlay is standalone. The base ZIP alone selects old official ClickBell and ItemsHandover. The local ItemsHandover freeze backup is not this formal overlay.

Inside the merged runtime, using official Linux/NVIDIA, Python3.11 and EmbodiChain0.2.4 plus dexsim-engine:

```bash
python verify_bundle.py
bash setup_policy.sh
bash eval.sh items_handover random 0 --max_episodes 100 --seed 0
```

The positional0 is GPU index. Replace `items_handover` with any task in [tasks.json](https://github.com/L-dot-eng/DHU-RoboLab-RoboSynChallenge-2026/blob/main/tasks.json). [README-submit.md](https://github.com/L-dot-eng/DHU-RoboLab-RoboSynChallenge-2026/blob/main/README-submit.md) lists all ten checkpoints and commands. Preserve shared VLM/tokenizer, all normalizers, RGB predictors, assets and dependencies. Inference is local-only/offline; first environment installation may need network.

HF archives contain complete source and inference artifacts. GitHub is a reviewable core-source mirror, NOT a standalone runtime.

## ItemsHandover selection and evidence

The selected teammate PredMask_ItemsHandover_50K uses unchanged SmolVLA weights and DeepLabV3-ResNet50 TorchScript segmentation of top/left-wrist/right-wrist RGB. Threshold0.70, red foreground, gray128 background, original float precision. ImageNet normalization is already embedded in the predictor. Fourteen robot state/action values and gripper6/13 scaling are preserved; stale six-dimensional metadata must not truncate state.

Internal50-step prediction/execution queue is preserved. The original caller requests10 actions at a time; it does not discard40 actions or replan every10. Reset occurs only between episodes. No consensus50, execution50, new grasping rules or retraining is enabled.

Our completed original-code cloud reproduction was10/30=33.3%; teammate history34/100 is a separate experiment. Interrupted trials6/23 and6/18 did not establish improvement, so we retain the fully measured original. These are public/local historical scores, NOT official hidden-test scores or a unified ten-task benchmark.

According to the teammate's documentation, policy fine-tuning used1000successful official-simulator episodes for50000steps. Simulator instance labels supervise the segmentation model only; policy input uses RGB-predicted masks, task instruction and robot proprioception. Runtime reads neither simulator masks nor object poses.

Portable-adapter validation: nine focused tests independently pass, including real TorchScript/original-encoder parity and the actual shared worker request loop with vendored queue methods. Five existing unified-entry tests also pass. Reviewer verified bitwise mask parity on full480x640 RGBA float/integer inputs. Neural chunk generation is mocked in the queue test. The cloud was off during integration: a new complete VLM/GPU/physics run of this portable entry was NOT performed.

Original measurement used NoMaterial robot visual assets and teammate predmask_eval. Both actual configs use350steps, but formal random retains top-camera focal_x/y perturbation±50 that the original measured config removed. We do not remove official perturbations and cannot promise33.3% under them. See [handover validation](https://github.com/L-dot-eng/DHU-RoboLab-RoboSynChallenge-2026/tree/main/validation/handover33) and [restore details](https://github.com/L-dot-eng/DHU-RoboLab-RoboSynChallenge-2026/blob/main/README-handover-update.md).

## Other selected policies and limitations

ClickBell84 retains the team's50K red/gray SmolVLA and RGB SegFormer-B0 predictor. After100base actions, up to five bounded50-action local retries use RGB visibility, robot kinematics and action history, not bell ground truth. Its independent100public scenes yield84successes/16failures under timeout-first scoring; raw85includes one simultaneous timeout. It is not a strict paired12-point causal gain over72. The portable entry previously passed six targeted complete cloud episodes; these are interface evidence, not another success-rate estimate. See [ClickBell84 validation](https://github.com/L-dot-eng/DHU-RoboLab-RoboSynChallenge-2026/tree/main/validation/clickbell84).

Drawer65/100, basket54/100 and pipette49/100 are their respective historical public results. The remaining five tasks retain official SmolVLA: table rearrangement, water pouring, mixer operating, item assembly, SampleLoading. Do not average heterogeneous results as one new benchmark or promise private-evaluation performance.

Preserve [THIRD_PARTY_NOTICES.md](https://github.com/L-dot-eng/DHU-RoboLab-RoboSynChallenge-2026/blob/main/THIRD_PARTY_NOTICES.md), including NVIDIA SegFormer's separate research/evaluation non-commercial terms and other upstream licenses. The root code license does not override model terms. No training trajectories, credentials, SSH keys or conversation records are included.
