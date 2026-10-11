"""Predicted RGB masks and the unchanged old 50K VLA; no hidden scene inputs."""
from collections import deque
import json
from pathlib import Path
import numpy as np


class ActionQueue:
    def __init__(self, infer):
        self.infer = infer
        self.actions = deque()
    def reset(self):
        self.actions.clear()
    def next(self, observation):
        if not self.actions:
            chunk = np.asarray(self.infer(observation), np.float32)
            if chunk.shape != (50,14) or not np.isfinite(chunk).all():
                raise ValueError('Expected 50 finite 14-dimensional actions')
            self.actions.extend(chunk)
        return self.actions.popleft().copy()


def mask_encoded_observation(encoded, predictor):
    keys = [f'observation.images.camera{i}' for i in (1,2,3)]
    images = [np.ascontiguousarray(np.moveaxis(encoded[key],0,-1)) for key in keys]
    result = dict(encoded)
    for key, image in zip(keys, predictor.enhance(images)):
        result[key] = np.ascontiguousarray(np.moveaxis(image,-1,0))
    return result


class Model:
    def __init__(self, checkpoint, vision, worker_args=None):
        from policy.robosyn_team.runtime.deploy_policy import SmolVLAWorkerClient
        from .vision import Predictor
        config = json.loads((Path(checkpoint)/'config.json').read_text())
        if (config['chunk_size'],config['n_action_steps']) != (50,50):
            raise ValueError('Policy prediction and execution must both be 50')
        args = dict(worker_args or {})
        args.update(checkpoint_dir=str(checkpoint),smolvla_steps=50,smolvla_rescale_gripper=True)
        args.setdefault('pytorch_device','cuda')
        self.predictor = Predictor(vision,device=args['pytorch_device'])
        self.client = SmolVLAWorkerClient(args)
        expected = {'observation.images.camera1':'cam_high','observation.images.camera2':'cam_left_wrist','observation.images.camera3':'cam_right_wrist'}
        if self.client.image_keys != expected:
            self.client.close()
            raise ValueError('Old checkpoint camera mapping differs')
        self.queue = ActionQueue(self.infer)
        self.instruction = None
        self.last_masks = None

    def infer(self, observation):
        from policy.robosyn_team.runtime.deploy_policy import encode_obs
        encoded = encode_obs(observation,self.client.image_keys)
        if encoded['observation.state'].shape != (14,):
            raise ValueError('Expected actual 14-joint observation despite stale metadata')
        masked = mask_encoded_observation(encoded,self.predictor)
        self.last_masks = [np.moveaxis(masked[f'observation.images.camera{i}'],0,-1) for i in (1,2,3)]
        return self.client.infer(masked,self.instruction)

    def action(self, obs, env):
        from policy.robosyn_team.runtime.deploy_policy import encode_action
        self.instruction = getattr(env,'_current_instruction',None)
        if not self.instruction:
            raise ValueError('Current task language instruction missing')
        return encode_action(self.queue.next(obs),env,rescale_gripper=True)

    def reset(self):
        self.queue.reset(); self.client.reset(); self.last_masks = None

    def close(self):
        self.client.close()
