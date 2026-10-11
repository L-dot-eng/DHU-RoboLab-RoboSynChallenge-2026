"""Original 10-request/50-internal-action RGB PredMask policy adapter."""
import time

import numpy as np

from ..runtime.deploy_policy import _extract_joint, _select_env, _to_numpy, encode_action
from .observations import apply_predicted_redgray_mask, load_segmenter


class Model:
    def __init__(self, vision_dir, worker_args):
        from ..runtime.deploy_policy import SmolVLAWorkerClient
        self.device = worker_args.get('pytorch_device', 'cuda')
        self.segmenter, self.threshold = load_segmenter(vision_dir, self.device)
        args = dict(worker_args, smolvla_steps=10)
        self.client = SmolVLAWorkerClient(args)

    def encode_obs(self, obs):
        state = _extract_joint(obs, 'qpos')
        encoded = {
            'observation.state': state,
            'observation.qvel': _extract_joint(obs, 'qvel', fallback=state),
            'observation.qf': _extract_joint(obs, 'qf', fallback=state),
        }
        for key, camera in self.client.image_keys.items():
            # The original modifier writes in place. Own this clone and read
            # only RGB, never simulator mask fields or object observations.
            color = obs['sensor'][camera]['color'].clone()
            masked = apply_predicted_redgray_mask(
                self, color, self.segmenter, self.threshold, self.device)
            image = _select_env(_to_numpy(masked))
            if image.ndim != 3:
                raise ValueError(f'Expected image for {camera} to be 3D, got {image.shape}')
            if image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
                chw = image
            else:
                chw = np.moveaxis(image, -1, 0)
            if chw.shape[0] == 1:
                chw = np.repeat(chw, 3, axis=0)
            elif chw.shape[0] > 3:
                chw = chw[:3]
            if np.issubdtype(chw.dtype, np.floating):
                chw = chw.astype(np.float32, copy=False)
                if float(np.nanmax(chw)) > 1.5:
                    chw = chw / 255.0
            else:
                chw = chw.astype(np.float32) / 255.0
            encoded[key] = np.ascontiguousarray(np.clip(chw, 0.0, 1.0))
        return encoded

    def eval(self, env, obs):
        instruction = getattr(env, '_current_instruction', None)
        if not instruction:
            raise RuntimeError('Task instruction is missing from gym_config')
        started = time.perf_counter()
        actions = self.client.infer(self.encode_obs(obs), task=instruction)
        compute_seconds = time.perf_counter() - started
        info = {}
        truncated = False
        for action in actions:
            action_tensor = encode_action(
                action, env, rescale_gripper=self.client.rescale_gripper,
                gripper_indices=self.client.gripper_indices,
                gripper_source_range=self.client.gripper_source_range)
            obs, _, _, truncated, info = env.step(action_tensor)
            if bool(_to_numpy(truncated).reshape(-1)[0]):
                break
        return obs, info, truncated, [compute_seconds]

    def reset(self):
        self.client.reset()

    def close(self):
        self.client.close()
