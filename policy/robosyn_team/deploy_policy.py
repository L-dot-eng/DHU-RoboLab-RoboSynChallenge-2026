"""Official four-return entry; model inputs are RGB and robot proprioception only."""
from collections import deque
import json
import os
from pathlib import Path
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def resolve_task(task_name, root=ROOT):
    root = Path(root).resolve()
    tasks = json.loads((root/'tasks.json').read_text(encoding='utf-8'))
    if task_name not in tasks:
        raise ValueError(f'Unknown task: {task_name}')
    spec = dict(tasks[task_name])
    for key in ('checkpoint','vision'):
        if key in spec:
            path = (root/spec[key]).resolve()
            if not path.is_relative_to(root):
                raise ValueError('Checkpoint path must stay within submission bundle')
            spec[key] = str(path)
    return spec


class RawModel:
    def __init__(self, worker_args):
        from .runtime.deploy_policy import SmolVLAWorkerClient
        self.client = SmolVLAWorkerClient(worker_args)
        self.queue = deque()

    def action(self, obs, env):
        from .runtime.deploy_policy import encode_obs, encode_action
        if not self.queue:
            task = getattr(env,'_current_instruction',None)
            if not task:
                raise ValueError('Task language instruction is missing')
            chunk = np.asarray(self.client.infer(encode_obs(obs,self.client.image_keys),task),np.float32)
            if chunk.shape != (50,14) or not np.isfinite(chunk).all():
                raise ValueError('Expected a finite 50x14 SmolVLA chunk')
            self.queue.extend(chunk)
        return encode_action(self.queue.popleft(),env,rescale_gripper=self.client.rescale_gripper)

    def reset(self):
        self.queue.clear()
        self.client.reset()

    def close(self):
        self.client.close()


def get_model(usr_args):
    root = Path(usr_args.get('bundle_root') or ROOT).resolve()
    task_name = usr_args['task_name']
    spec = resolve_task(task_name,root)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['ROBOSYN_COMMON_VLM'] = str(root/'common/SmolVLM2-500M-Video-Instruct')
    worker_args = {
        'checkpoint_dir':spec['checkpoint'],
        'lerobot_root':str(root/'vendor/lerobot'),
        'smolvla_steps':50,
        'pytorch_device':usr_args.get('pytorch_device','cuda'),
        'smolvla_cuda_visible_devices':usr_args.get('gpu_id',0),
        'smolvla_rescale_gripper':True,
    }
    if usr_args.get('smolvla_python'):
        worker_args['smolvla_python']=usr_args['smolvla_python']
    if spec['kind']=='official_smolvla':
        return RawModel(worker_args)
    if spec['kind']=='pipette9':
        from .pipette.adapter import Model
        return Model(spec['checkpoint'],spec['vision'],worker_args,spec['instruction'])
    if spec['kind']=='predicted_mask':
        from .masked.adapter import Model
        return Model(spec['checkpoint'],spec['vision'],task_name,
                     os.environ['ROBOSYN_COMMON_VLM'],device=worker_args['pytorch_device'])
    raise ValueError(f"Unsupported model kind: {spec['kind']}")


def eval(env, model, obs):
    compute_seconds=0.
    info={}
    truncated=False
    for _ in range(50):
        started=time.perf_counter()
        action=model.action(obs,env)
        compute_seconds+=time.perf_counter()-started
        obs,_,terminated,truncated,info=env.step(action)
        done_truncated = bool(truncated.any().item()) if hasattr(truncated,'any') else bool(truncated)
        done_terminated = bool(terminated.any().item()) if hasattr(terminated,'any') else bool(terminated)
        if done_truncated or done_terminated:
            break
        if env.get_wrapper_attr('is_task_success')():
            break
    return obs,info,truncated,[compute_seconds]


def reset_model(model):
    model.reset()


def close_model(model):
    model.close()
