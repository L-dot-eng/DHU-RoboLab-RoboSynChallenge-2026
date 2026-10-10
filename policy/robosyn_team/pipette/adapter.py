"""RGB prediction only, full 50-action chunks, learned gripper commands."""
# Adapted from project-018/work/deploy.py and the selected depth9_residual.py.
# Historical evaluation bookkeeping is deliberately outside this adapter.
from collections import deque
import json
from pathlib import Path
import numpy as np
from .pipette_contract import CAMERAS, check_action_chunk
from .depth_residual import depth_offset, accepted_solution, lower_left


class ActionQueue:
    def __init__(self,infer):self.infer=infer; self.actions=deque()
    def reset(self):self.actions.clear()
    def next(self,obs):
        if not self.actions:
            chunk=np.asarray(self.infer(obs),np.float32); check_action_chunk(chunk)
            self.actions.extend(chunk)
        return self.actions.popleft().copy()


class Model:
    def __init__(self,checkpoint,vision,worker_args,task):
        cfg=json.loads((Path(checkpoint)/'config.json').read_text())
        if (cfg['chunk_size'],cfg['n_action_steps'],cfg['num_steps'])!=(50,50,10):raise ValueError('Deployment horizon mismatch')
        from .vision import Predictor
        from policy.robosyn_team.runtime.deploy_policy import SmolVLAWorkerClient
        args=dict(worker_args)
        args.update(checkpoint_dir=str(checkpoint),smolvla_steps=50,smolvla_rescale_gripper=True)
        args.setdefault('pytorch_device','cuda')
        self.predictor=Predictor(vision,device=args['pytorch_device'])
        self.task=task
        self.client=SmolVLAWorkerClient(args)
        self.queue=ActionQueue(self.infer)
        self.step_index=0;self.accepted=0;self.rejected=0

    def infer(self,obs):
        from policy.robosyn_team.runtime.deploy_policy import encode_obs
        encoded=encode_obs(obs)
        images=[np.moveaxis(encoded['observation.images.'+c],0,-1) for c in CAMERAS]
        for c,image in zip(CAMERAS,self.predictor.enhance(images)):
            encoded['observation.images.'+c]=np.ascontiguousarray(np.moveaxis(image,-1,0))
        return self.client.infer(encoded,self.task)

    def action(self,obs,env):
        from policy.robosyn_team.runtime.deploy_policy import encode_action
        action=encode_action(self.queue.next(obs),env,rescale_gripper=True)
        offset=depth_offset(self.step_index);self.step_index+=1
        if offset:
            robot_env=getattr(env,'unwrapped',env)
            action,accepted=lower_left(action,robot_env.robot,robot_env.single_action_space.low,robot_env.single_action_space.high,offset)
            self.accepted+=int(accepted);self.rejected+=int(not accepted)
        return action

    def reset(self):
        self.step_index=0;self.accepted=0;self.rejected=0
        self.queue.reset();self.client.reset()

    def close(self):self.client.close()
