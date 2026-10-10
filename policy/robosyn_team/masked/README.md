# Selected drawer and basket adapters

`Model(checkpoint, vision, task_name, common_vlm, device='cuda')` supports
`drawer_open_place` and `handle_basket`, with `action(obs, env)`, `reset()`, and
`close()`. `checkpoint` and `vision` are direct local model directories;
`common_vlm` is the packaged local SmolVLM directory. `action` returns a CPU
float32 tensor of shape `[1,14]`, matching the recovered adapters before
`env.step`.

The selected versions are the drawer recovery-5k handle predictor (historical
65/100) and basket contact-v2 policy (historical 54/100). These are prior measured
results, not new measurements of this portable package. Source functions are
retained from the recovered `base_work`/`run_work` and `base_work`/`contact_v2`
inference paths. The copied source files had no copyright notices; no new rights
to model weights or third-party dependencies are implied.

| Behavior | Drawer | Basket |
| --- | --- | --- |
| Camera order | high, left wrist, right wrist | high, right wrist, left wrist |
| Policy image keys | camera1, camera2, camera3 | cam_high, cam_right_wrist, cam_left_wrist |
| Resize to 384x288 | OpenCV INTER_LINEAR | OpenCV INTER_AREA |
| Foreground overlays | original 25% palette; handle 25% purple | original 25% palette; contact labels 45% |
| State grippers | normalized using live robot limits | same formula, preserving float64 limits conversion |
| Output grippers | original denormalize | original physical_actions |

Both retain `predict_action_chunk`, per-action saved postprocessing, all 50
actions per prediction, 10 denoising steps, original task text, and reset
semantics. Raw camera RGB comes from the live simulator's camera `color` channel;
only the robot's own active joint positions and limits are otherwise read.
Object poses, segmentation truth, expert controllers, and expert action
trajectories are not accessed. Live gripper limits must remain `[0,0.05]`.

SegFormer uses the saved image processor and local-only model loading. SmolVLA
loads its saved configuration, retains `load_vlm_weights`, redirects the VLM and
tokenizer to `common_vlm`, and loads the checkpoint with `strict=True`.
Historical training-directory and evaluation-record checks are removed; the
outer bundle manifest owns file-integrity validation. No old absolute paths are
used. `close()` clears model and processor references for task switching.

Run `python -m pytest policy/robosyn_team/masked/test_masked_adapter.py -q` from
the package root. Set `MASKED_REFERENCE_ROOT` to the directory containing
`recovered_drawer` and `recovered_basket` for source comparison tests. That path
is used only by tests. PyTorch and OpenCV tests skip if those dependencies are
missing. Controlled substitutes in the unit tests verify boundaries and do not
establish real model inference or task success; those require the cloud checks.
