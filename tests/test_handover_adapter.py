"""CPU regression checks; original-source parity is optional via environment."""
import importlib
import importlib.util
import ast
from collections import deque
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'policy'))


class Segmenter(torch.nn.Module):
    def forward(self, x):
        return {'out': (x[:, :1] - 0.5) * 10}


def observation(dtype=torch.float32):
    color = torch.tensor([[[[0.0, 0.2, 0.3, 0.9], [1.0, 0.4, 0.2, 0.7]],
                           [[0.5, 0.4, 0.2, 0.6], [0.9, 0.1, 0.3, 0.8]]]])
    if dtype == torch.uint8:
        color = (color * 255).to(dtype)
    return {'robot': {'qpos': torch.arange(14).reshape(1, 14).float()},
            'sensor': {name: {'color': color.clone()} for name in
                       ('cam_high', 'cam_left_wrist', 'cam_right_wrist')}}


class FakeEnv:
    def __init__(self, truncated_at=None):
        self.unwrapped = self
        self.device = 'cpu'
        self._current_instruction = 'handover instruction'
        self.single_action_space = SimpleNamespace(
            shape=(14,), low=np.zeros(14), high=np.ones(14) * 0.04)
        self.actions = []
        self.truncated_at = truncated_at

    def step(self, action):
        self.actions.append(action.clone())
        return {'step': len(self.actions)}, 0, torch.tensor([True]), torch.tensor(
            [len(self.actions) == self.truncated_at]), {'step': len(self.actions)}

    def get_wrapper_attr(self, name):
        return lambda: True


class FakeClient:
    def __init__(self, args):
        self.smolvla_steps = args['smolvla_steps']
        self.image_keys = {'observation.images.camera1': 'cam_high',
                           'observation.images.camera2': 'cam_left_wrist',
                           'observation.images.camera3': 'cam_right_wrist'}
        self.rescale_gripper = True
        self.gripper_indices = (6, 13)
        self.gripper_source_range = (0.0, 1.0)
        self.requests = []
        self.next_action = 0

    def infer(self, obs, task):
        self.requests.append((obs, task, self.smolvla_steps))
        actions = [np.full(14, idx, np.float32) for idx in
                   range(self.next_action, self.next_action + self.smolvla_steps)]
        self.next_action += self.smolvla_steps
        return actions

    def reset(self):
        self.next_action = 0

    def close(self):
        pass


class HandoverTests(unittest.TestCase):
    def setUp(self):
        try:
            self.adapter = importlib.import_module('robosyn_team.handover.adapter')
        except ModuleNotFoundError:
            self.fail('RGB-only handover adapter has not been implemented')
        self.tmp = tempfile.TemporaryDirectory(prefix='handover_', dir=ROOT / 'tests')
        self.addCleanup(self.tmp.cleanup)
        self.vision = Path(self.tmp.name)
        with (self.vision / 'best_model.ts').open('wb') as stream:
            torch.jit.save(torch.jit.trace(Segmenter(), torch.zeros(1, 3, 2, 2), strict=False), stream)
        (self.vision / 'threshold.json').write_text('{"threshold": 0.7}')

    def model(self):
        with patch('robosyn_team.runtime.deploy_policy.SmolVLAWorkerClient', FakeClient):
            return self.adapter.Model(str(self.vision), {'smolvla_steps': 50,
                                                       'pytorch_device': 'cpu'})

    def test_mask_float_precision_clone_and_all_three_cameras(self):
        obs = observation()
        original = {key: value['color'].clone() for key, value in obs['sensor'].items()}
        encoded = self.model().encode_obs(obs)
        self.assertEqual(encoded['observation.state'].shape, (14,))
        for key in ('observation.images.camera1', 'observation.images.camera2',
                    'observation.images.camera3'):
            self.assertEqual(encoded[key].dtype, np.float32)
            np.testing.assert_array_equal(encoded[key][:, 0, 1], [1., 0., 0.])
            np.testing.assert_array_equal(encoded[key][:, 0, 0],
                                          np.full(3, np.float32(128) * np.float32(1 / 255)))
        for key in original:
            torch.testing.assert_close(obs['sensor'][key]['color'], original[key], rtol=0, atol=0)

    def test_integer_rgb_and_missing_velocity_force(self):
        encoded = self.model().encode_obs(observation(torch.uint8))
        self.assertEqual(encoded['observation.images.camera1'].dtype, np.float32)
        np.testing.assert_array_equal(encoded['observation.qvel'], np.zeros(14))
        np.testing.assert_array_equal(encoded['observation.qf'], np.zeros(14))
        np.testing.assert_array_equal(encoded['observation.images.camera1'][:, 0, 0],
                                      np.full(3, np.float32(128 / 255)))

    def test_only_reads_rgb_and_robot_observations(self):
        class Forbidden:
            def __array__(self, *args, **kwargs):
                raise AssertionError('ground truth read')
        obs = observation()
        obs['object_poses'] = Forbidden()
        for camera in obs['sensor'].values():
            camera['segmentation'] = Forbidden()
        self.assertEqual(self.model().encode_obs(obs)['observation.state'].shape, (14,))

    def test_eval_ten_actions_ignores_termination_and_success(self):
        model = self.model()
        env = FakeEnv()
        from robosyn_team.deploy_policy import eval
        for _ in range(5):
            result = eval(env, model, observation())
            self.assertEqual(len(result), 4)
            self.assertEqual(len(result[3]), 1)
            self.assertGreaterEqual(result[3][0], 0)
        self.assertEqual(len(env.actions), 50)
        self.assertEqual([entry[2] for entry in model.client.requests], [10] * 5)
        self.assertEqual([float(action[0, 0]) for action in env.actions], list(range(50)))
        self.assertAlmostEqual(float(env.actions[1][0, 6]), 0.04, places=7)

    def test_truncation_stops_batch_reset_discards_remaining_worker_actions(self):
        model = self.model()
        env = FakeEnv(truncated_at=3)
        from robosyn_team.deploy_policy import eval, reset_model
        result = eval(env, model, observation())
        self.assertEqual(len(env.actions), 3)
        self.assertTrue(bool(result[2][0]))
        reset_model(model)
        new_env = FakeEnv()
        eval(new_env, model, observation())
        self.assertEqual(float(new_env.actions[0][0, 0]), 0.)

    def test_dispatch_only_handover_requests_ten(self):
        from robosyn_team.deploy_policy import get_model, resolve_task
        with patch('robosyn_team.runtime.deploy_policy.SmolVLAWorkerClient', FakeClient):
            model = get_model({'task_name': 'items_handover', 'pytorch_device': 'cpu'})
            self.assertIsInstance(model, self.adapter.Model)
            self.assertEqual(model.client.smolvla_steps, 10)
            raw = get_model({'task_name': 'table_rearrangement', 'pytorch_device': 'cpu'})
            self.assertEqual(raw.client.smolvla_steps, 50)
        spec = resolve_task('items_handover')
        self.assertEqual(Path(spec['checkpoint']).name, 'items_handover_33.3pct_ours')

    def test_other_task_eval_retains_success_break(self):
        from robosyn_team.deploy_policy import eval
        env = FakeEnv()
        model = SimpleNamespace(action=lambda obs, env: torch.zeros(1, 14))
        eval(env, model, {})
        self.assertEqual(len(env.actions), 1)

    def test_real_shared_worker_keeps_fifty_queue_across_ten_requests_and_reset(self):
        # Execute the vendored methods themselves, avoiding heavyweight VLM
        # imports. Only neural chunk generation and processors are replaced.
        vendor = ROOT / 'vendor/lerobot/src/lerobot/policies'
        tree = ast.parse((vendor / 'smolvla/modeling_smolvla.py').read_text(encoding='utf-8'))
        policy_class = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                            and node.name == 'SmolVLAPolicy')
        names = {'reset', 'select_action', '_check_get_actions_condition', '_rtc_enabled', '_prepare_batch'}
        methods = [node for node in policy_class.body if isinstance(node, ast.FunctionDef)
                   and node.name in names]
        utils = ast.parse((vendor / 'utils.py').read_text(encoding='utf-8'))
        populate = next(node for node in utils.body if isinstance(node, ast.FunctionDef)
                        and node.name == 'populate_queues')
        namespace = {'torch': torch, 'deque': deque, 'ACTION': 'action', 'OBS_STATE': 'observation.state'}
        source = 'from __future__ import annotations\n' + ast.unparse(ast.Module(
            body=[populate, *methods], type_ignores=[]))
        exec(compile(source, '<vendored queue methods>', 'exec'), namespace)
        QueuePolicy = type('QueuePolicy', (), {name: namespace[name] for name in names})
        policy = QueuePolicy()
        policy.config = SimpleNamespace(n_action_steps=50, rtc_config=None, adapt_to_pi_aloha=False)
        policy.eval = lambda: None
        generated = []
        def generate(batch, noise=None):
            start = 50 * len(generated)
            generated.append(batch)
            return torch.arange(start, start + 50).float().reshape(1, 50, 1).expand(1, 50, 14)
        policy._get_action_chunk = generate
        policy.reset()
        from robosyn_team.runtime import smolvla_worker as worker
        obs_path = self.vision / 'obs.npz'
        np.savez(obs_path, **{'observation.state': np.arange(14, dtype=np.float32),
                             'task': np.asarray('handover instruction')})
        infer = {'cmd': 'infer', 'obs_path': str(obs_path), 'n_action_steps': 10}
        requests = [infer] * 6 + [{'cmd': 'reset'}, infer]
        stdout = io.StringIO()
        with patch.object(worker, '_parse_args', return_value=SimpleNamespace()), \
             patch.object(worker, '_load_runtime', return_value=(policy, lambda obs: obs, lambda action: action)), \
             patch.object(sys, 'stdin', io.StringIO('\n'.join(map(json.dumps, requests)) + '\n')), \
             patch.object(sys, 'stdout', stdout):
            self.assertEqual(worker.main(), 0)
        responses = [json.loads(line) for line in stdout.getvalue().splitlines()]
        self.assertTrue(all(response['ok'] for response in responses))
        first_fifty = [action[0] for response in responses[:5] for action in response['actions']]
        self.assertEqual(first_fifty, list(range(50)))
        self.assertEqual(responses[5]['actions'][0][0], 50.)
        self.assertEqual(responses[7]['actions'][0][0], 100.)
        self.assertEqual(len(generated), 3)
        self.assertEqual(len(policy._queues['action']), 40)
        self.assertEqual(generated[0]['observation.state'].shape, (14,))

    @unittest.skipUnless(os.environ.get('HANDOVER_ORIGINAL_SOURCE'), 'original source not supplied')
    def test_real_segmenter_and_encoding_match_frozen_original(self):
        original = Path(os.environ['HANDOVER_ORIGINAL_SOURCE'])
        def load(name, path):
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        mask = load('original_handover_mask', original / 'robosynchallenge/predmask/observations.py')
        encoder = load('original_handover_encoder', original / 'policy/smolvla/deploy_policy.py')
        vision = ROOT / 'checkpoints/items_handover_33.3pct_ours/segmentation'
        # Python file handles avoid Windows TorchScript Unicode path failures.
        with (vision / 'best_model.ts').open('rb') as stream:
            original_segmenter = torch.jit.load(stream, map_location='cpu').eval()
        threshold = json.loads((vision / 'threshold.json').read_text())['threshold']
        mask._RUNTIME_CACHE[(str((vision / 'best_model.ts').resolve()), 'cpu')] = (
            original_segmenter, threshold)
        with patch('robosyn_team.runtime.deploy_policy.SmolVLAWorkerClient', FakeClient):
            model = self.adapter.Model(str(vision), {'smolvla_steps': 50, 'pytorch_device': 'cpu'})
        torch.manual_seed(17)
        obs = observation()
        for camera in obs['sensor'].values():
            camera['color'] = torch.rand(1, 32, 48, 4)
        original_obs = {'robot': obs['robot'], 'sensor': {
            key: {'color': value['color'].clone()} for key, value in obs['sensor'].items()}}
        env = SimpleNamespace()
        for camera in original_obs['sensor'].values():
            mask.apply_predicted_redgray_mask(env, camera['color'], str(vision / 'best_model.ts'),
                                             str(vision / 'threshold.json'), device='cpu')
        expected = encoder.encode_obs(original_obs, model.client.image_keys)
        actual = model.encode_obs(obs)
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)


if __name__ == '__main__':
    unittest.main()
