"""Pure action/label contracts. Inference never imports the simulator labeler."""
# Inference-only subset of the recovered drawer/basket base contracts.
import numpy as np

GRIPPERS = (6, 13)


def normalize(action, limits):
    result = np.asarray(action, dtype=np.float32).copy()
    limits = np.asarray(limits)
    if result.shape != (14,) or not np.isfinite(result).all():
        raise ValueError('Expected 14 finite joint commands')
    if limits.shape != (2, 2) or np.any(limits[:, 1] <= limits[:, 0]):
        raise ValueError('Invalid gripper limits')
    result[list(GRIPPERS)] = (result[list(GRIPPERS)] - limits[:, 0]) / (limits[:, 1] - limits[:, 0])
    return result


def denormalize(action, limits):
    result = np.asarray(action, dtype=np.float32).copy()
    if result.shape != (14,) or not np.isfinite(result).all():
        raise ValueError('Expected 14 finite joint commands')
    limits = np.asarray(limits)
    result[list(GRIPPERS)] = np.clip(result[list(GRIPPERS)], 0, 1) * (limits[:, 1] - limits[:, 0]) + limits[:, 0]
    return result


def image_rgb(tensor):
    array = tensor[0, ..., :3].detach().cpu().numpy()
    if array.dtype.kind == 'f':
        array = np.clip(array * 255, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)
