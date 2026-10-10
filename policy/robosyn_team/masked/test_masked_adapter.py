"""Inference contract checks; controlled stubs are not task evaluations."""
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
    assert (HERE / 'adapter.py').is_file(), 'Portable drawer/basket adapter is missing'
    return importlib.import_module('policy.robosyn_team.masked.adapter')


@pytest.mark.parametrize('task', ['drawer_open_place', 'handle_basket'])
def test_full_chunk_next_reset_and_physical_grippers(task):
    module = adapter()
    model = module.Model.__new__(module.Model)
    model.kind = module.TASKS[task]
    calls = []

    def predict(observation):
        calls.append(observation)
        chunk = np.repeat(np.arange(50, dtype=np.float32)[:, None], 14, axis=1)
        chunk[:, 6], chunk[:, 13] = -.1, 1.1
        return chunk

    model.queue = model.kind.FullChunk(predict)
    model.policy = SimpleNamespace(reset=lambda: None)
    model.steps = 0
    for i in range(50):
        action = model.next('first')
        assert action[0] == i
        np.testing.assert_array_equal(action[[6, 13]], [0, 1])
    assert calls == ['first']
    model.next('second')
    model.reset()
    assert model.next('third')[0] == 0
    assert calls == ['first', 'second', 'third']
    assert model.steps == 1
    physical = model.kind.denormalize(action, [[0, .05], [0, .05]])
    np.testing.assert_allclose(physical[[6, 13]], [0, .05])
    np.testing.assert_array_equal(physical[[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]], 49)


@pytest.mark.parametrize('task', ['drawer_open_place', 'handle_basket'])
def test_bad_chunk_is_rejected_before_any_action(task):
    kind = adapter().TASKS[task]
    for chunk in (np.zeros((49, 14)), np.full((50, 14), np.inf)):
        with pytest.raises(ValueError):
            kind.FullChunk(lambda observation: chunk).next(None)


def test_gripper_normalization_preserves_other_observed_joints():
    module = adapter()
    state = np.arange(14, dtype=np.float32)
    state[[6, 13]] = [.025, .05]
    normalized = module.normalize(state, np.array([[0, .05], [0, .05]], np.float32))
    np.testing.assert_array_equal(normalized[[6, 13]], [.5, 1])
    np.testing.assert_array_equal(normalized[:6], state[:6])
    np.testing.assert_array_equal(normalized[7:13], state[7:13])
    np.testing.assert_array_equal(state[[6, 13]], np.array([.025, .05], np.float32))


def test_renderer_special_labels_keep_distinct_selected_policies():
    module = adapter()
    rgb = np.full((1, 2, 3), 100, np.uint8)
    drawer = module.TASKS['drawer_open_place'].enhance(rgb, np.array([[7, 3]], np.uint8))
    basket = module.TASKS['handle_basket'].enhance(rgb, np.array([[6, 5]], np.uint8))
    np.testing.assert_array_equal(drawer, [[[125, 91, 139], [100, 100, 100]]])
    np.testing.assert_array_equal(basket, [[[170, 84, 141], [100, 100, 100]]])


def test_policy_inputs_preserve_task_text_camera_keys_and_float_scaling():
    module = adapter()
    images = [np.full((288, 384, 3), value, np.uint8) for value in (51, 102, 153)]
    state = np.zeros(14, np.float32)
    drawer = module.TASKS['drawer_open_place'].policy_arrays(images, state)
    basket = module.TASKS['handle_basket'].policy_arrays(images, state)
    assert drawer['task'] == 'Open the drawer, pick the tomato with the other arm then place it in the drawer.'
    assert basket['task'] == 'Pick the bottle then place it to the basket, carry the basket with the other arm.'
    np.testing.assert_array_equal(drawer['observation.images.camera2'], np.float32(.4))
    np.testing.assert_array_equal(basket['observation.images.cam_right_wrist'], np.float32(.4))
    np.testing.assert_array_equal(basket['observation.images.cam_left_wrist'], np.float32(.6))


@pytest.mark.parametrize('task', ['drawer_open_place', 'handle_basket'])
def test_loader_uses_local_vlm_tokenizer_and_strict_checkpoint(monkeypatch, task):
    module = adapter()
    config = SimpleNamespace(chunk_size=50, n_action_steps=50, num_steps=10,
                             load_vlm_weights=True, vlm_model_name='remote', device='cuda')
    calls = {}

    class Policy:
        def __init__(self):
            self.config = config

        def to(self, device):
            calls['device'] = device
            return self

        def eval(self):
            return self

        def reset(self):
            pass

    def load_config(path, **kwargs):
        calls['config'] = (path, kwargs)
        return config

    def load_policy(path, **kwargs):
        calls['policy'] = (path, kwargs)
        return Policy()

    def processors(cfg, **kwargs):
        calls['processors'] = kwargs
        return None, None

    monkeypatch.setitem(sys.modules, 'lerobot.configs.policies',
                        SimpleNamespace(PreTrainedConfig=SimpleNamespace(from_pretrained=load_config)))
    monkeypatch.setitem(sys.modules, 'lerobot.policies.smolvla.modeling_smolvla',
                        SimpleNamespace(SmolVLAPolicy=SimpleNamespace(from_pretrained=load_policy)))
    monkeypatch.setitem(sys.modules, 'lerobot.policies.factory',
                        SimpleNamespace(make_pre_post_processors=processors))
    monkeypatch.setitem(sys.modules, 'policy.robosyn_team.masked.vision',
                        SimpleNamespace(Predictor=lambda folder, kind, device: (folder, kind, device)))
    model = module.Model('checkpoint', 'vision', task, 'local/common-vlm', device='cpu')
    assert config.vlm_model_name == 'local/common-vlm'
    assert config.load_vlm_weights is True
    assert config.device == 'cpu'
    assert calls['policy'][1] == {'config': config, 'local_files_only': True, 'strict': True}
    assert calls['processors']['preprocessor_overrides'] == {
        'device_processor': {'device': 'cpu'},
        'tokenizer_processor': {'tokenizer_name': 'local/common-vlm'},
    }
    assert model.predictor[0] == 'vision'
    model.close()
    assert model.policy is None and model.predictor is None


class ArrayTensor:
    """Only the read-only tensor operations used to obtain RGB/robot observations."""
    def __init__(self, array):
        self.array = np.asarray(array)

    def __getitem__(self, key):
        return ArrayTensor(self.array[key])

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array


def environment():
    state = np.zeros((1, 14), np.float32)
    state[:, [6, 13]] = [.025, .05]
    limits = np.repeat(np.array([[[0, .05]]], np.float32), 14, axis=1)
    sensors = {camera: ArrayTensor(np.full((1, 480, 640, 4), value, np.uint8))
               for camera, value in [('cam_high', 51), ('cam_left_wrist', 102), ('cam_right_wrist', 153)]}
    robot = SimpleNamespace(get_qpos=lambda: ArrayTensor(state),
                            get_joint_ids=lambda name, remove_mimic: [6 if name == 'left_eef' else 13],
                            body_data=SimpleNamespace(qpos_limits=ArrayTensor(limits)))
    return SimpleNamespace(active_joint_ids=list(range(14)), robot=robot,
                           sim=SimpleNamespace(get_sensor=lambda camera: SimpleNamespace(
                               get_data=lambda: {'color': sensors[camera]})))


@pytest.mark.parametrize('task,order,mode', [('drawer_open_place', [51, 102, 153], 1),
                                          ('handle_basket', [51, 153, 102], 3)])
def test_observed_reads_only_rgb_robot_state_and_uses_task_resize(monkeypatch, task, order, mode):
    module = adapter()
    model = module.Model.__new__(module.Model)
    model.kind = module.TASKS[task]
    resize_calls = []

    def resize(image, shape, interpolation):
        resize_calls.append((shape, interpolation))
        return np.full((288, 384, 3), image[0, 0, 0], np.uint8)

    monkeypatch.setitem(sys.modules, 'cv2', SimpleNamespace(resize=resize, INTER_LINEAR=1, INTER_AREA=3))
    observation, limits = model.observed(environment())
    assert [observation[c][0, 0, 0] for c in model.kind.CAMERAS] == order
    assert resize_calls == [((384, 288), mode)] * 3
    np.testing.assert_allclose(limits, [[0, .05], [0, .05]])
    np.testing.assert_array_equal(observation['state'][[6, 13]], [.5, 1])


def selected_functions(path, names, namespace):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


@pytest.mark.skipif(not os.environ.get('MASKED_REFERENCE_ROOT'), reason='Recovered references not supplied')
@pytest.mark.parametrize('task', ['drawer_open_place', 'handle_basket'])
def test_pixels_and_policy_arrays_match_recovered_sources(task):
    module = adapter()
    root = Path(os.environ['MASKED_REFERENCE_ROOT'])
    kind = module.TASKS[task]
    namespace = {'np': np, 'CLASSES': kind.CLASSES, 'PALETTE': kind.PALETTE,
                 'CAMERAS': kind.CAMERAS}
    if task == 'drawer_open_place':
        root = root / 'recovered_drawer/source'
        selected_functions(root / 'base_work/vision.py', ['enhance_rgb'], namespace)
        selected_functions(root / 'run_work/refine_vision.py', ['enhance'], namespace)
        selected_functions(root / 'base_work/evaluate.py', ['policy_observation'], namespace)
        expected_arrays = namespace['policy_observation']
    else:
        root = root / 'recovered_basket/source'
        selected_functions(root / 'contact_v2/contact_vision.py', ['enhance_rgb'], namespace)
        namespace['enhance'] = namespace['enhance_rgb']
        namespace['TASK'] = 'Pick the bottle then place it to the basket, carry the basket with the other arm.'
        selected_functions(root / 'base_work/contracts.py', ['policy_arrays'], namespace)
        expected_arrays = namespace['policy_arrays']
    rng = np.random.default_rng(54)
    rgb = rng.integers(0, 256, (288, 384, 3), dtype=np.uint8)
    mask = rng.integers(0, 8, (288, 384), dtype=np.uint8)
    enhanced = kind.enhance(rgb, mask)
    np.testing.assert_array_equal(enhanced, namespace['enhance'](rgb, mask))
    state = rng.random(14, dtype=np.float32)
    actual, expected = kind.policy_arrays([enhanced] * 3, state), expected_arrays([enhanced] * 3, state)
    assert actual.keys() == expected.keys()
    for key in actual:
        np.testing.assert_array_equal(actual[key], expected[key])


@pytest.mark.parametrize('task', ['drawer_open_place', 'handle_basket'])
def test_real_torch_action_has_physical_grippers_and_batch_dimension(task):
    module = adapter()
    torch = pytest.importorskip('torch')
    pytest.importorskip('cv2')
    model = module.Model.__new__(module.Model)
    model.kind = module.TASKS[task]
    model.steps = 0
    chunk = np.full((50, 14), .4, np.float32)
    model.queue = model.kind.FullChunk(lambda _: chunk)
    output = model.action({}, environment())
    assert isinstance(output, torch.Tensor) and tuple(output.shape) == (1, 14)
    np.testing.assert_allclose(output[0, [6, 13]].cpu().numpy(), [.02, .02])
    np.testing.assert_array_equal(output[0, :6].cpu().numpy(), np.float32(.4))


def test_official_wrapper_only_exposes_unwrapped_robot():
    module=adapter()
    torch=pytest.importorskip('torch')
    pytest.importorskip('cv2')
    model=module.Model.__new__(module.Model)
    model.kind=module.TASKS['handle_basket']
    model.steps=0
    model.queue=model.kind.FullChunk(lambda _:np.full((50,14),.4,np.float32))
    wrapped=SimpleNamespace(unwrapped=environment())
    output=model.action({},wrapped)
    assert tuple(output.shape)==(1,14)


@pytest.mark.parametrize('task', ['drawer_open_place', 'handle_basket'])
def test_real_torch_predict_postprocesses_every_action_in_order(task):
    module = adapter()
    torch = pytest.importorskip('torch')
    model = module.Model.__new__(module.Model)
    model.kind = module.TASKS[task]
    model.steps = 0
    model.replan_steps = []
    model.predictor = SimpleNamespace(enhance=lambda images: images)
    chunk = torch.arange(700, dtype=torch.float32).reshape(1, 50, 14) / 100
    model.policy = SimpleNamespace(predict_action_chunk=lambda batch: chunk)
    model.pre = lambda batch: batch
    model.post = lambda action: action * 3 - 1
    observation = {camera: np.full((288, 384, 3), value, np.uint8)
                   for camera, value in zip(model.kind.CAMERAS, (51, 102, 153))}
    observation['state'] = np.zeros(14, np.float32)
    actions = model.predict(observation)
    expected = np.arange(700, dtype=np.float32).reshape(50, 14) / 100 * 3 - 1
    np.testing.assert_allclose(actions, expected, rtol=1e-6)
    assert model.replan_steps == [0]
    assert [image[0, 0, 0] for image in model.last_inputs] == [51, 102, 153]


@pytest.mark.parametrize('task', ['drawer_open_place', 'handle_basket'])
def test_real_cv2_resize_matches_selected_original_on_nonuniform_pixels(task):
    module = adapter()
    cv2 = pytest.importorskip('cv2')
    model = module.Model.__new__(module.Model)
    model.kind = module.TASKS[task]
    rng = np.random.default_rng(65)
    rgb = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
    env = environment()
    env.sim = SimpleNamespace(get_sensor=lambda _: SimpleNamespace(
        get_data=lambda: {'color': ArrayTensor(rgb[None])}))
    observation, _ = model.observed(env)
    if task == 'drawer_open_place':
        expected = cv2.resize(rgb, (384, 288))
    else:
        expected = cv2.resize(rgb, (384, 288), interpolation=cv2.INTER_AREA)
    for camera in model.kind.CAMERAS:
        np.testing.assert_array_equal(observation[camera], expected)
