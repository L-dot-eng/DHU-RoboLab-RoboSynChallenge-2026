"""Shared RGB-only rendering and episode/action contracts."""
# Inference subset of project-018/work/pipette_contract.py.
import numpy as np

CAMERAS = ('cam_high', 'cam_right_wrist', 'cam_left_wrist')
LABELS = ('background', 'pipette_body', 'pipette_button', 'beaker')
COLORS = np.array([[0, 0, 0], [0, 180, 255], [255, 0, 0], [255, 210, 0]], np.float32)


def overlay(rgb, labels, confidence):
    if (rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3
            or labels.shape != rgb.shape[:2] or confidence.shape != labels.shape
            or not np.issubdtype(labels.dtype, np.integer) or labels.min() < 0 or labels.max() > 3
            or not np.isfinite(confidence).all() or confidence.min() < 0 or confidence.max() > 1):
        raise ValueError('Invalid RGB, labels or confidence')
    result = rgb.copy()
    accepted = (labels > 0) & (confidence >= .6)
    result[accepted] = np.rint(.75 * rgb[accepted] + .25 * COLORS[labels[accepted]]).astype(np.uint8)
    return result


def check_action_chunk(actions):
    if actions.shape != (50, 14) or not np.isfinite(actions).all():
        raise ValueError('Expected 50 finite 14-dimensional actions')
