# ItemsHandover v4 validation and limits

## Frozen original evidence

`original30-evaluation_metrics.json` is an unchanged extract of the completed original-code cloud run on 2026-10-11: 10 successes, 30 complete episodes. The first startup-only failed attempt is not counted. Neither interrupted execution50 (6/23) nor consensus50 (6/18) is included. No weight training occurred during those execution experiments.

Original policy SHA256: 7d298746aa2b0cdd9060aea9a855e142f602b3ce3061535b7d4b045493365ec8.
Original TorchScript predictor SHA256: 8a2aa42546165c681001dacfa60e626a834f3d2c69875fb1f4d75bc133107f33.

Original cloud: Python3.11, torch2.10.0+cu128, EmbodiChain0.2.4, dexsim-engine0.4.3, RTX4090, NoMaterial robot visual assets. No claim of new clean-environment or hidden-scene performance. Historical teammate34/100 is a separate experiment.

## Portable-adapter checks performed 2026-10-11

Nine new regression tests were run independently by implementer, reviewer and release owner: all nine pass, none skipped when the original source is supplied. Local environment: Windows, Python3.13.15, torch2.9.1+cpu, NumPy2.3.5.

Coverage includes:

- Real supplied TorchScript predictor and original-source encoder output parity, float128/255 gray preserved, three camera mapping, input tensors not mutated.
- Reviewer additionally verified bitwise mask equality on full480x640 synthetic RGBA inputs in both float32 and uint8, with alpha preserved.
- Full14D state/action and gripper6/13 scaling. The true normalizer state statistics are14D, despite stale6D input metadata. Original and existing shared-compatible policy have the same500 tensor names/shapes.
- Original10-action outer batch, original truncated-only early return, unchanged other-task success/termination behavior, episode reset.
- Actual shared worker JSON request loop plus vendored queue/reset/select_action methods: five requests consume one50-action chunk, sixth generates the next chunk; reset discards remaining queued actions. Neural chunk generation and processor calls in this queue test are mocked; this is not a full VLA forward pass.
- RGB and ordinary robot state only; simulator segmentation and object-pose observation extras are ignored.

Re-run the focused suite from the bundle root with torch installed:

```bash
python -m unittest discover -s tests -p test_handover_adapter.py -v
```

Set `HANDOVER_ORIGINAL_SOURCE` to the extracted teammate code root for the original-source parity case; otherwise that optional case is skipped. The original comparison source is retained in the local freeze, not duplicated in this inference overlay.

An attempted broader local pytest run reported24 passed,2 skipped(CUDA unavailable),2 failed because the local CPU environment lacks the external `draccus` dependency required to import LeRobot configs. This is not a blanket claim that all upstream tests passed. The supplied Linux deployment dependency specification is unchanged.

## Independent scope review

Read-only review found no production-code blocker. Existing shared worker/runtime, vendor code, other nine task controllers, official evaluator, scene configs and their success conditions are unchanged. Routing adds only the Handover branch and its dedicated eval path. Model files match original hashes. Root checksums cover the assembled release; the incremental ZIP was read back against file hashes.

## Explicit limits

The cloud had been shut down before this portable adapter was built. Full strict VLM loading, new GPU inference and new complete physics episodes through the final unified entry were NOT performed. Real segmenter tests and queue mocks do not replace those checks; prior original30 evidence is not relabelled as a portable-entry acceptance run.

The active official `configs/items_handover/random/gym_config.json` and teammate predmask_eval both use350-step limits and matching base cameras/distractor settings. Official random retains high-camera focal_x/y perturbations±50, while predmask_eval removes them. The formal release does not remove those official perturbations. Therefore10/30 is not a promised score for formal random or private evaluation.
