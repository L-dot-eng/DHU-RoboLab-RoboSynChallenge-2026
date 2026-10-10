"""Portable adapter contracts; these tests do not measure task success."""
import ast
import importlib
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))


def adapter():
    assert (HERE / 'adapter.py').is_file(), 'Portable 9 mm adapter is missing'
    return importlib.import_module('policy.robosyn_team.pipette.adapter')


def test_press_offset_has_exact_9mm_schedule():
    module = adapter()
    for step, expected in [(0, 0), (199, 0), (200, 0), (201, .000042048),
                           (205, .000936), (210, .003168), (225, .009),
                           (249, .009), (250, .009), (270, .000936),
                           (274, .000042048), (275, 0), (999, 0)]:
        assert module.depth_offset(step) == pytest.approx(expected, abs=1e-15)


def test_queue_consumes_all_fifty_actions_and_reset_discards_remainder():
    module = adapter()
    calls = []

    def infer(obs):
        calls.append(obs)
        return np.repeat(np.arange(50)[:, None], 14, axis=1)

    queue = module.ActionQueue(infer)
    for index in range(50):
        assert queue.next('first')[0] == index
    assert calls == ['first']
    assert queue.next('second')[0] == 0
    queue.reset()
    assert queue.next('third')[0] == 0
    assert calls == ['first', 'second', 'third']


@pytest.mark.parametrize('chunk', [np.zeros((49, 14)), np.full((50, 14), np.nan)])
def test_queue_rejects_invalid_chunk_before_returning_action(chunk):
    queue = adapter().ActionQueue(lambda _: chunk)
    with pytest.raises(ValueError):
        queue.next(None)


def test_overlay_rounding_background_and_confidence_boundary():
    adapter()
    from policy.robosyn_team.pipette.pipette_contract import overlay
    rgb = np.full((2, 3, 3), 100, np.uint8)
    labels = np.array([[0, 1, 2], [3, 1, 2]], np.uint8)
    confidence = np.array([[1, .6, 1], [1, .599, 0]], np.float32)
    result = overlay(rgb, labels, confidence)
    np.testing.assert_array_equal(result, [
        [[100, 100, 100], [75, 120, 139], [139, 75, 75]],
        [[139, 128, 75], [100, 100, 100], [100, 100, 100]],
    ])
    np.testing.assert_array_equal(rgb, np.full((2, 3, 3), 100, np.uint8))


def test_joint_guard_rejects_large_nonfinite_out_of_bounds_and_wrong_shape():
    guard = adapter().accepted_solution
    seed = np.zeros(6)
    low, high = -np.ones(6), np.ones(6)
    assert guard(seed, np.full(6, .12), low, high)
    for candidate in (np.full(6, .12001), np.full(6, np.nan), np.zeros((1, 6))):
        assert not guard(seed, candidate, low, high)
    assert not guard(np.full(6, .99), np.full(6, 1.01), low, high)


def test_reset_clears_phase_and_pending_actions_without_reading_episode_files(monkeypatch):
    module = adapter()
    model = module.Model.__new__(module.Model)
    model.queue = module.ActionQueue(lambda _: np.zeros((50, 14)))
    model.queue.next(None)
    model.step_index, model.accepted, model.rejected = 243, 30, 12
    reset_calls = []
    model.client = SimpleNamespace(reset=lambda: reset_calls.append(True), close=lambda: None)

    def forbidden_read(*args, **kwargs):
        raise AssertionError('Reset/close tried to access an external episode file')

    monkeypatch.setattr(Path, 'read_text', forbidden_read)
    model.reset()
    model.close()
    assert (model.step_index, model.accepted, model.rejected) == (0, 0, 0)
    assert not model.queue.actions
    assert reset_calls == [True]


def test_first_action_has_no_eval_directory_dependency(monkeypatch):
    module = adapter()
    chunk = np.repeat(np.arange(50)[:, None], 14, axis=1)
    model = module.Model.__new__(module.Model)
    model.queue = module.ActionQueue(lambda _: chunk)
    model.step_index = model.accepted = model.rejected = 0
    runtime_name = 'policy.robosyn_team.runtime.deploy_policy'

    def encode_action(action, env, rescale_gripper):
        assert rescale_gripper is True
        return action.reshape(1, 14)

    monkeypatch.setitem(sys.modules, runtime_name, SimpleNamespace(encode_action=encode_action))

    def forbidden_read(*args, **kwargs):
        raise AssertionError('Action tried to access an external episode file')

    monkeypatch.setattr(Path, 'read_text', forbidden_read)
    np.testing.assert_array_equal(model.action({}, None), np.zeros((1, 14)))
    assert model.step_index == 1


@pytest.mark.parametrize('wrapped',[False,True])
def test_residual_uses_zero_based_action_index_and_only_press_phase(monkeypatch,wrapped):
    module = adapter()
    model = module.Model.__new__(module.Model)
    model.queue = module.ActionQueue(lambda _: np.zeros((50, 14)))
    model.step_index = model.accepted = model.rejected = 0
    monkeypatch.setitem(sys.modules, 'policy.robosyn_team.runtime.deploy_policy',
                        SimpleNamespace(encode_action=lambda action, env, rescale_gripper: action[None, :]))

    def lower_left(action, robot, low, high, offset):
        corrected = action.copy()
        corrected[:, 2] -= offset
        return corrected, True

    monkeypatch.setattr(module, 'lower_left', lower_left)
    env = SimpleNamespace(robot=None, single_action_space=SimpleNamespace(low=None, high=None))
    if wrapped: env=SimpleNamespace(unwrapped=env)
    actions = np.concatenate([model.action({}, env) for _ in range(276)])
    np.testing.assert_array_equal(actions[:201], 0)
    np.testing.assert_array_equal(actions[275:], 0)
    assert actions[201, 2] == pytest.approx(-.000042048)
    assert actions[225, 2] == pytest.approx(-.009)
    assert actions[274, 2] == pytest.approx(-.000042048)
    assert (model.step_index, model.accepted, model.rejected) == (276, 74, 0)


def test_constructor_uses_supplied_paths_task_and_fixed_horizon(monkeypatch):
    module = adapter()
    reads, worker_calls = [], []

    def read_config(path, *args, **kwargs):
        reads.append(path)
        return '{"chunk_size":50,"n_action_steps":50,"num_steps":10}'

    monkeypatch.setattr(Path, 'read_text', read_config)
    monkeypatch.setitem(sys.modules, 'policy.robosyn_team.pipette.vision',
                        SimpleNamespace(Predictor=lambda folder, device: (folder, device)))
    monkeypatch.setitem(sys.modules, 'policy.robosyn_team.runtime.deploy_policy',
                        SimpleNamespace(SmolVLAWorkerClient=lambda args: worker_calls.append(args)))
    args = {'lerobot_root': 'vendor/lerobot', 'pytorch_device': 'cpu', 'smolvla_steps': 1}
    model = module.Model('relocated/checkpoint', 'relocated/vision', args, 'pipette task')
    assert reads == [Path('relocated/checkpoint/config.json')]
    assert model.task == 'pipette task'
    assert model.predictor == ('relocated/vision', 'cpu')
    assert worker_calls[0]['checkpoint_dir'] == 'relocated/checkpoint'
    assert worker_calls[0]['smolvla_steps'] == 50
    assert worker_calls[0]['smolvla_rescale_gripper'] is True
    assert args['smolvla_steps'] == 1


def test_constructor_rejects_changed_horizon_before_loading_models(monkeypatch):
    module = adapter()
    monkeypatch.setattr(Path, 'read_text',
                        lambda *args, **kwargs: '{"chunk_size":50,"n_action_steps":10,"num_steps":10}')
    with pytest.raises(ValueError, match='horizon'):
        module.Model('checkpoint', 'vision', {}, 'task')


def reference_functions(path, names, namespace):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


@pytest.mark.skipif(not os.environ.get('PIPETTE_REFERENCE_DIR'), reason='Legacy reference not supplied')
def test_numpy_outputs_match_original_selected_49pct_sources():
    module = adapter()
    from policy.robosyn_team.pipette.pipette_contract import overlay, COLORS
    root = Path(os.environ['PIPETTE_REFERENCE_DIR'])
    original6 = reference_functions(root / 'depth_residual.py', ['depth_offset'], {})['depth_offset']
    original9 = reference_functions(root / 'depth9_residual.py', ['depth_offset'],
                                    {'original_offset': original6})['depth_offset']
    assert [module.depth_offset(i) for i in range(1200)] == [original9(i) for i in range(1200)]
    original_overlay = reference_functions(root / 'pipette_contract.py', ['overlay'],
                                           {'np': np, 'COLORS': COLORS})['overlay']
    rng = np.random.default_rng(912)
    rgb = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
    labels = rng.integers(0, 4, (480, 640), dtype=np.uint8)
    confidence = rng.random((480, 640), dtype=np.float32)
    np.testing.assert_array_equal(overlay(rgb, labels, confidence), original_overlay(rgb, labels, confidence))


@pytest.mark.parametrize('failure', [None, 'invalid', 'large', 'position', 'rotation', 'none'])
def test_torch_ik_only_changes_left_joints_and_falls_back_on_failed_checks(failure):
    module = adapter()
    torch = pytest.importorskip('torch')

    class Robot:
        def __init__(self):
            self.calls = 0

        def compute_fk(self, q, name, to_matrix):
            assert name == 'left_arm' and to_matrix
            self.calls += 1
            pose = torch.eye(4).unsqueeze(0)
            pose[:, :3, 3] = q[:, :3]
            if self.calls == 2 and failure == 'position':
                pose[:, 0, 3] += .001
            if self.calls == 2 and failure == 'rotation':
                pose[:, 0, 0] += .003
            return pose

        def compute_ik(self, target, joint_seed, name):
            assert name == 'left_arm'
            if failure == 'none':
                return None
            candidate = joint_seed.clone()
            candidate[:, :3] = target[:, :3, 3]
            if failure == 'large':
                candidate[:, 3] += .13
            return torch.tensor([failure != 'invalid']), candidate

    action = torch.arange(14, dtype=torch.float32).reshape(1, 14) / 100
    corrected, accepted = module.lower_left(action, Robot(), -np.ones(14), np.ones(14), .009)
    assert accepted is (failure is None)
    assert torch.equal(corrected[:, 6:], action[:, 6:])
    if failure is None:
        assert corrected[0, 2].item() == pytest.approx(.011)
        assert action[0, 2].item() == pytest.approx(.02)
    else:
        assert torch.equal(corrected, action)
