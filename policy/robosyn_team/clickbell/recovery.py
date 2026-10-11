"""Finite RGB-gated local TCP retry. Lowest measured TCP is only a contact heuristic."""
from collections import deque
import numpy as np
from .adapter import Model as BaselineModel

OFFSETS = ((0., 0.), (.004, 0.), (-.004, 0.), (0., .004), (0., -.004))
CAMERAS = ('cam_high', 'cam_left_wrist', 'cam_right_wrist')


def as_numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def accepted_solution(reference, candidate, low, high):
    candidate = np.asarray(candidate)
    return bool(candidate.shape == (6,) and np.isfinite(candidate).all()
                and np.max(np.abs(candidate - reference)) <= .12
                and np.all(candidate >= low) and np.all(candidate <= high))


def plan_attempt(anchor, offset, current, previous, solve):
    """Validate all three IK targets and all 50 actions before returning any action."""
    if tuple(offset) not in OFFSETS:
        raise ValueError('Offset outside the five approved local candidates')
    current = np.asarray(current, np.float32)
    previous = np.asarray(previous, np.float32)
    if current.shape != (6,) or previous.shape != (14,) or not np.isfinite(previous).all() or not np.isfinite(current).all():
        return None, 'invalid_state_or_action'
    approach = np.asarray(anchor).copy()
    approach[:2, 3] += offset
    approach[2, 3] += .006
    press = approach.copy(); press[2, 3] -= .009
    joints = []
    # Every local IK remains close to the measured contact candidate.
    for target in (approach, press, approach):
        candidate = solve(target, current)
        if candidate is None:
            return None, 'ik_rejected'
        candidate = np.asarray(candidate, np.float32)
        if candidate.shape != (6,) or not np.isfinite(candidate).all():
            return None, 'invalid_ik_shape_or_values'
        joints.append(candidate)
    result = np.tile(previous, (50, 1))
    cursor = 0; start = current
    for target, length in ((joints[0], 20), (joints[1], 12), (joints[1], 8), (joints[2], 10)):
        result[cursor:cursor + length, 7:13] = np.linspace(start, target, length + 1)[1:]
        cursor += length; start = target
    differences = np.diff(np.vstack([previous, result]), axis=0)[:, 7:13]
    if not np.isfinite(result).all() or np.max(np.abs(differences)) > .04:
        return None, 'joint_step_exceeds_0.04'
    return result, None


class Kinematics:
    def __init__(self, robot):
        self.robot = robot
        limits = as_numpy(robot.get_qpos_limits(name='right_arm'))[0]
        if limits.shape != (6, 2) or not np.isfinite(limits).all():
            raise ValueError('Expected finite right-arm limits (1,6,2)')
        self.low, self.high = limits[:, 0], limits[:, 1]

    def tensor(self, array):
        import torch
        return torch.as_tensor(array, dtype=torch.float32, device=self.robot.device)

    def fk(self, q):
        pose = as_numpy(self.robot.compute_fk(self.tensor(q).reshape(1, 6), name='right_arm', to_matrix=True))[0]
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError('Invalid robot FK matrix')
        return pose.copy()

    def solve(self, target, reference):
        solution = self.robot.compute_ik(self.tensor(target)[None], joint_seed=self.tensor(reference)[None], name='right_arm')
        if solution is None:
            return None
        valid, candidate = solution
        if not bool(np.all(as_numpy(valid))):
            return None
        candidate = as_numpy(candidate).reshape(-1)
        if not accepted_solution(reference, candidate, self.low, self.high):
            return None
        checked = self.fk(candidate)
        if np.linalg.norm(checked[:3, 3] - target[:3, 3]) > .0005 or np.max(np.abs(checked[:3, :3] - target[:3, :3])) > .002:
            return None
        return candidate


class Model:
    def __init__(self, checkpoint, vision, worker_args=None):
        self.baseline = BaselineModel(checkpoint, vision, worker_args)
        self._clear()

    def _clear(self):
        self.step = 0
        self.history = []
        self.recovery_events = []
        self.actions = deque()
        self.attempt = 0
        self.disabled = False
        self.anchor = None
        self.reference = None
        self.last_action = None

    @property
    def last_masks(self):
        return self.baseline.last_masks

    def reset(self):
        self.baseline.reset()
        self._clear()

    def close(self):
        self.baseline.close()

    def _event(self, **event):
        self.recovery_events.append({'step': self.step, **event})

    def _prepare(self, obs, env):
        sensors = obs.get('sensor', {})
        if any(camera not in sensors or 'color' not in sensors[camera] for camera in CAMERAS):
            self._event(kind='skip', reason='missing_rgb'); self.disabled = True
            return
        from policy.robosyn_team.runtime.deploy_policy import _extract_image
        images = [np.moveaxis(_extract_image(obs, c), 0, -1) for c in CAMERAS]
        areas = [int(np.count_nonzero(mask == 2)) for mask in self.baseline.predictor.predict(images)]
        self._event(kind='visibility', bell_pixels=areas)
        if max(areas, default=0) < 16:
            self._event(kind='skip', reason='bell_not_visible'); self.disabled = True
            return
        kin = Kinematics(env.robot)
        if self.anchor is None:
            poses = [(index, q, kin.fk(q)) for index, q in self.history]
            if not poses:
                self._event(kind='skip', reason='missing_history'); self.disabled = True
                return
            index, self.reference, self.anchor = min(poses, key=lambda row: row[2][2, 3])
            self._event(kind='selected', history_index=index, measured_qpos=self.reference.tolist(), pose=self.anchor.tolist())
        current = as_numpy(obs['robot']['qpos']).reshape(-1)[7:13].copy()
        offset = OFFSETS[self.attempt]; self.attempt += 1
        # IK seeds and locality bounds use the measured candidate, not a drifting retry.
        planned, reason = plan_attempt(self.anchor, offset, current, as_numpy(self.last_action).reshape(-1), lambda target, unused: kin.solve(target, self.reference))
        self._event(kind='attempt', attempt=self.attempt, offset=list(offset), accepted=planned is not None, reason=reason)
        if planned is None:
            self.disabled = True
            return
        self.actions.extend(planned)

    def action(self, obs, env):
        if 20 <= self.step <= 85:
            q = as_numpy(obs['robot']['qpos']).reshape(-1)[7:13].copy()
            self.history.append((self.step, q))
        if self.step >= 100 and not self.disabled and not self.actions and self.attempt < 5:
            self._prepare(obs, env)
        if self.actions:
            import torch
            action = torch.as_tensor(self.actions.popleft(), dtype=torch.float32, device=self.last_action.device)[None]
            if not self.actions:
                self._event(kind='attempt_complete', attempt=self.attempt)
        else:
            action = self.baseline.action(obs, env)
        self.last_action = action
        self.step += 1
        return action
