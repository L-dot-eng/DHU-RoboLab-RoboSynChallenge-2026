# Fixed 9 mm pipette adapter

`Model(checkpoint, vision, worker_args, task)` in `adapter.py` exposes
`action(obs, env)`, `reset()`, and `close()` for the selected pipette policy.
The caller supplies the pretrained-model directory, the SegFormer directory,
the shared SmolVLA worker arguments, and the original task text.

The source is project-018's `deploy.py`, `vision.py`, `pipette_contract.py`,
`depth_residual.py`, and `depth9_residual.py`. Their inference behavior is
retained; paths and imports are adapted, and historical evaluation recording
is omitted. Source module docstrings are retained. No source copyright notice
was present in those five files; this adaptation does not grant additional
rights to the models or bundled third-party runtime.

- The checkpoint must declare chunk size 50, action horizon 50, and 10 denoising
  steps. All 50 actions are consumed before a new prediction.
- RGB input uses the four-class SegFormer predictor, confidence threshold 0.6,
  alpha 0.25, and the original color palette and rounding.
- The fixed smooth press window is action indices 200 through 274, zero based;
  its first value is zero and its maximum is 0.009 m.
- Only the left six joints can change. The original joint limit, 0.12 rad
  deviation, 0.5 mm position residual, and 0.002 rotation-entry checks remain.
- Reset clears pending actions, phase, counters, and worker state. No episode
  cursor, evaluation output directory, task metadata file, or old project
  location is required at runtime.

The main process needs PyTorch, NumPy, and Transformers for SegFormer and FK/IK.
The shared worker has its own packaged SmolVLA dependencies. SegFormer loading
uses `local_files_only=True`; all supplied weight/configuration files must
already exist. `worker_args['pytorch_device']` selects the main predictor and
worker device, defaulting to CUDA.

Run `python -m pytest policy/robosyn_team/pipette/test_adapter.py -q` from the
package root. Optionally set `PIPETTE_REFERENCE_DIR` to the original project-018
work directory for direct curve and renderer comparisons. This test-only
reference is not a deployment dependency. PyTorch checks skip when PyTorch is
unavailable; those skips do not establish FK/IK or model inference correctness.
The tests include controlled dependency substitutes and do not measure task
success. The historical 49/100 result belongs to the original evaluated model;
it is not a new evaluation of this package.
