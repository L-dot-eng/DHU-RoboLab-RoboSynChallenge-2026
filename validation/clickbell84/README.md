# ClickBell84 evidence and limits

## Independent public evaluation

`heldout100.json` records the frozen bounded-recovery candidate on100 unique feasible randomized scenes, sequential candidates starting at9300000. The raw harness counted85task-predicate successes. Official `scripts/eval_policy.py` counts success only when NOT truncated: applying that timeout-first rule yields84successes/16failures,0initial successes, unchanged physical horizon361. `official-score.json` preserves the derived score and the one excluded collision. The raw log is deliberately not rewritten. Official expert feasibility/reset and task predicate remain unchanged. The runtime controller does not access feasibility results, object ground truth, task success or button displacement; the external harness records these for diagnostics only.

-77successes occurred in the first100 physical actions.
-7successes occurred while a local recovery attempt was executing.
-1additional task-predicate success occurred at step361, after all five attempts had finished and the base policy resumed; timeout was also true. Seed9300041 is an official failure, not one of the84successes.

The last category is not a valid official success. The12percentage-point difference versus the72/100 baseline is not a proven causal gain. Development pairs had different initial image hashes even with nominally identical scene seeds; fixed identical RGB/state inputs did yield identical seeded VLA chunks. This held-out set is a separate absolute public-performance estimate with timeout precedence corrected from recorded flags, not a matched-pair experiment or official hidden-test result. It was not pooled with development episodes.

The100-run harness initializes through `embodichain.utils.set_seed(0)` and seeds worker inference per episode. The final portable entry retains the existing official-compatible no-argument worker reset rather than the experimental seeded wrapper. Exact per-seed actions/results are therefore not promised by the portable entry. Source/model/config hashes and this protocol are retained in `heldout100.json`; code and model file integrity for the delivered artifact is separately covered by root `checksums.json`.

## Actual portable entry acceptance

`unified-smoke.json` and `unified-smoke.log` record execution of the packaged `get_model`, four-return `eval`, `reset_model` and `close_model` interfaces, with official feasibility/reset and361physical-step horizon. The final recovery module SHA256 is `c76a8843474f962fb3f4d03020594122a542e5948f1e37917cd7241bceee5ca1`.

The bounded test tried six existing development scenes. Five completed successfully and one timed out normally. These6selected scenes are interface coverage, NOT another unbiased success-rate estimate. Two episodes naturally reached accepted recovery attempts: seed9200010 succeeded at127steps; seed9200011 executed all five attempts and timed out at361steps. In both, first recovery began at controller step100 with attempt1. All six episode resets and a final reset cleared the baseline action queue, recovery queue, step/attempt counters, history, cached masks and anchor. Entry actions were finite(1,14), and controller/physical action counts matched throughout. The process exited normally.

Local checks:26controller/port tests passed;5new integration/smoke-driver tests passed; packaged authored tests38passed/20skipped, with skipped dependencies documented in local integration records. The local machine lacks the full LeRobot/GPU environment; these results are not a claim that every vendored upstream test passed. GPU acceptance ran on the existing cloud installation with local-only model loading.

The portable controller differs from the experimental one only by relative baseline import and no-argument reset. It reuses the exact72% policy/predictor files; no new VLA training or hidden-state controller input was introduced. The lowest-TCP target is a heuristic, not an image-derived exact object pose.

Existing cloud uses NoMaterial robot visual assets and an installed simulation environment. This is not a clean-from-empty-system verification and does not guarantee performance under standard unseen assets or private competition perturbations. Other nine task files/routes remain unchanged from the original submission.
