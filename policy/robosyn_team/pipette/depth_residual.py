"""Bounded press-phase experiment, using only robot kinematics, never object state."""
# Kinematics from project-018/work/depth_residual.py; fixed 9 mm depth9 schedule.
import numpy as np


def depth_offset(step):
    if step<200 or step>=275:return 0.
    t=min(1.,(step-200)/25.,(275-step)/25.)
    return min(.009,1.5*(.006*t*t*(3-2*t)))


def accepted_solution(seed,candidate,low,high):
    candidate=np.asarray(candidate)
    return bool(candidate.shape==(6,) and np.isfinite(candidate).all()
        and np.max(np.abs(candidate-seed))<=.12 and np.all(candidate>=low) and np.all(candidate<=high))


def lower_left(action,robot,low,high,offset):
    import torch
    if not offset:return action,False
    q=action[:,:6];pose=robot.compute_fk(q,name='left_arm',to_matrix=True)
    target=pose.clone();target[:,2,3]-=offset
    solution=robot.compute_ik(target,joint_seed=q,name='left_arm')
    if solution is None:return action,False
    valid,candidate=solution
    if not bool(valid.all()):return action,False
    candidate=candidate.reshape(1,6)
    if not accepted_solution(q[0].detach().cpu().numpy(),candidate[0].detach().cpu().numpy(),low[:6],high[:6]):return action,False
    checked=robot.compute_fk(candidate,name='left_arm',to_matrix=True)
    if (torch.linalg.vector_norm(checked[:,:3,3]-target[:,:3,3]).item()>.0005
            or torch.max(torch.abs(checked[:,:3,:3]-target[:,:3,:3])).item()>.002):return action,False
    corrected=action.clone();corrected[:,:6]=candidate
    if not torch.equal(corrected[:,6:],action[:,6:]):raise AssertionError('Non-left-arm action changed')
    return corrected,True
