"""RGB/proprioception-only SmolVLA adapter; complete 50-action chunks."""
# Inference functions from recovered_drawer base_work and run_work sources.
import numpy as np
from .contracts import denormalize

CAMERAS = ('cam_high', 'cam_left_wrist', 'cam_right_wrist')
CLASSES = ('background', 'object', 'drawer_moving', 'cabinet', 'left_gripper', 'right_gripper', 'robot', 'handle')
PALETTE = {1: (255,64,64), 2: (64,128,255), 4: (64,220,128), 5: (255,192,64)}
INTERPOLATION = 'INTER_LINEAR'


def enhance_rgb(rgb, mask):
    rgb, mask = np.asarray(rgb), np.asarray(mask)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3 or mask.shape != rgb.shape[:2]:
        raise ValueError('Expected uint8 RGB HWC and aligned label HW')
    if mask.dtype.kind not in 'iu' or mask.size == 0 or mask.min() < 0 or mask.max() >= len(CLASSES):
        raise ValueError('Unknown or noninteger semantic label')
    output = rgb.astype(np.float32)
    gray = output @ np.array([.299,.587,.114], dtype=np.float32)
    background = mask == 0
    output[background] = .5 * output[background] + .5 * gray[background, None]
    for label, color in PALETTE.items():
        selected = mask == label
        output[selected] = .75 * output[selected] + .25 * np.array(color, dtype=np.float32)
    return np.clip(np.rint(output), 0, 255).astype(np.uint8)


def enhance(rgb,mask):
    output=enhance_rgb(rgb,mask)
    selected=mask==7
    output[selected]=np.clip(np.rint(.75*rgb[selected]+.25*np.array([200,64,255])),0,255).astype(np.uint8)
    return output


def checked_observation(observation):
    if set(observation) != {*CAMERAS, 'state'}:
        raise ValueError('Only named RGB cameras and observed robot state are allowed')
    images = [np.asarray(observation[c]) for c in CAMERAS]
    state = np.asarray(observation['state'], dtype=np.float32)
    if state.shape != (14,) or not np.isfinite(state).all():
        raise ValueError('Expected 14 finite observed joints')
    if any(im.shape != (288,384,3) or im.dtype != np.uint8 for im in images):
        raise ValueError('Expected ordered uint8 RGB cameras, each 384x288')
    return images, state.copy()


def policy_observation(images, state, predictor=None):
    import numpy as np
    if len(images) != 3 or np.asarray(state).shape != (14,) or not np.isfinite(state).all():
        raise ValueError('Expected three cameras and 14 finite observed joints')
    inputs = predictor.enhance(images) if predictor else images
    if len(inputs) != 3 or any(im.dtype != np.uint8 or im.shape != (288,384,3) for im in inputs):
        raise ValueError('Invalid policy image contract')
    batch = {'observation.state': np.asarray(state, dtype=np.float32).copy(),
             'task': 'Open the drawer, pick the tomato with the other arm then place it in the drawer.'}
    for index, rgb in enumerate(inputs, 1):
        batch[f'observation.images.camera{index}'] = rgb.copy().transpose(2,0,1).astype(np.float32)/255
    return batch


policy_arrays = policy_observation


class FullChunk:
    """Execute every action in a 50-step prediction exactly once."""
    def __init__(self, predict):
        self.predict = predict
        self.reset()

    def reset(self):
        self.actions = None
        self.index = 50

    def next(self, observation):
        if self.index == 50:
            actions = np.asarray(self.predict(observation), dtype=np.float32)
            if actions.shape != (50,14) or not np.isfinite(actions).all():
                raise ValueError('Expected 50 finite 14-joint predicted actions')
            self.actions = actions.copy()
            self.actions[:, [6,13]] = np.clip(self.actions[:, [6,13]],0,1)
            self.index = 0
        action = self.actions[self.index].copy()
        self.index += 1
        return action
