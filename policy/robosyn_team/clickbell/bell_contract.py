"""Deployment-only palette. No simulator annotations or scene state."""
import numpy as np

LABELS = ('background', 'robot', 'bell')


def render_redgray(labels):
    labels = np.asarray(labels)
    if (labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer)
            or labels.size == 0 or labels.min() < 0 or labels.max() > 2):
        raise ValueError('Expected a nonempty H,W integer label map with classes 0..2')
    result = np.full((*labels.shape, 3), 128, np.uint8)
    result[labels > 0] = (255, 0, 0)
    return result
