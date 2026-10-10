"""Portable deployment of the selected drawer-65 and basket-54 RGB policies."""
import numpy as np
from . import drawer, basket
from .contracts import image_rgb, normalize

TASKS = {'drawer_open_place': drawer, 'handle_basket': basket}


class Model:
    def __init__(self, checkpoint, vision, task_name, common_vlm, device='cuda'):
        self.kind = TASKS[task_name]
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.policies.factory import make_pre_post_processors
        from .vision import Predictor
        config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
        if (config.chunk_size, config.n_action_steps, config.num_steps) != (50,50,10):
            raise ValueError('Selected policy must predict and execute full 50 actions')
        config.vlm_model_name = str(common_vlm)
        config.device = device
        self.policy = SmolVLAPolicy.from_pretrained(checkpoint, config=config,
                                                    local_files_only=True, strict=True).to(device).eval()
        self.pre, self.post = make_pre_post_processors(self.policy.config, pretrained_path=checkpoint,
            preprocessor_overrides={'device_processor': {'device': device},
                                    'tokenizer_processor': {'tokenizer_name': str(common_vlm)}})
        self.predictor = Predictor(vision, self.kind, device=device)
        self.queue = self.kind.FullChunk(self.predict)
        self.reset()

    def predict(self, observation):
        import torch
        if self.kind is drawer:
            images, state = drawer.checked_observation(observation)
            arrays = drawer.policy_observation(images, state, self.predictor)
            self.last_inputs = [np.rint(arrays[f'observation.images.camera{i}'].transpose(1,2,0)*255).astype(np.uint8)
                                for i in (1,2,3)]
        else:
            if set(observation) != set(basket.CAMERAS) | {'state'}:
                raise ValueError('Unexpected contact policy observation channel')
            images = [observation[camera] for camera in basket.CAMERAS]
            self.last_inputs = self.predictor.enhance(images)
            arrays = basket.policy_arrays(self.last_inputs, observation['state'])
        batch = {key: torch.from_numpy(value) if isinstance(value, np.ndarray) else value
                 for key, value in arrays.items()}
        with torch.inference_mode():
            chunk = self.policy.predict_action_chunk(self.pre(batch))
            if tuple(chunk.shape) != (1,50,14):
                raise ValueError('Policy returned an unexpected chunk')
            actions = torch.stack([self.post(chunk[:,i,:])[0] for i in range(50)]).cpu().numpy()
        if actions.shape != (50,14) or not np.isfinite(actions).all():
            raise ValueError('Invalid postprocessed actions')
        self.raw_chunk = actions.copy()
        self.replan_steps.append(self.steps)
        return self.raw_chunk

    def next(self, observation):
        action = self.queue.next(observation)
        self.steps += 1
        if self.kind is basket:
            action[[6,13]] = np.clip(action[[6,13]],0,1)
        return action

    def observed(self, env):
        import cv2
        env = getattr(env,'unwrapped',env)
        active = list(env.active_joint_ids)
        grip = [int(env.robot.get_joint_ids(name=name, remove_mimic=True)[0])
                for name in ('left_eef','right_eef')]
        if len(active) != 14 or [active[6], active[13]] != grip:
            raise ValueError('Robot action layout changed')
        limits = env.robot.body_data.qpos_limits[0,grip].cpu().numpy()
        if self.kind is basket:
            limits = np.asarray(limits, dtype=np.float64)
        if (limits.shape != (2,2) or not np.isfinite(limits).all()
                or not np.allclose(limits, [[0,.05],[0,.05]], atol=1e-6, rtol=0)):
            raise ValueError('Live gripper limits differ from the selected policy')
        observation = {camera: cv2.resize(image_rgb(env.sim.get_sensor(camera).get_data()['color']),
                                         (384,288), interpolation=getattr(cv2, self.kind.INTERPOLATION))
                       for camera in self.kind.CAMERAS}
        observation['state'] = normalize(env.robot.get_qpos()[0,active].cpu().numpy(), limits)
        if self.kind is drawer:
            drawer.checked_observation(observation)
        return observation, limits

    def action(self, obs, env):
        import torch
        observation, limits = self.observed(env)
        canonical = self.next(observation)
        physical = self.kind.denormalize(canonical, limits)
        device = getattr(getattr(env,'unwrapped',env),'device','cpu')
        return torch.as_tensor(physical, dtype=torch.float32, device=device).unsqueeze(0)

    def reset(self):
        self.policy.reset()
        self.queue.reset()
        self.steps = 0
        self.replan_steps = []
        self.last_inputs = None
        self.raw_chunk = None

    def close(self):
        self.queue.reset()
        self.policy = self.predictor = self.pre = self.post = None
        self.last_inputs = self.raw_chunk = None
