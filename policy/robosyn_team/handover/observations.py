"""Original teammate RGB-only TorchScript masking arithmetic."""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch


def apply_predicted_redgray_mask(env, color, model, threshold, device):
    if str(device).startswith('cuda') and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    rgb = color[..., :3]
    single = rgb.ndim == 3
    batch = rgb.unsqueeze(0) if single else rgb
    batch = batch.to(device=device, dtype=torch.float32)
    if float(batch.max()) > 1.5:
        batch = batch / 255.0
    if batch.shape[-1] == 3:
        batch = batch.permute(0, 3, 1, 2)
    elif batch.shape[1] != 3:
        raise ValueError(f'Expected RGB NHWC or NCHW tensor, got {tuple(batch.shape)}')
    with torch.inference_mode():
        logits = model(batch)
        if isinstance(logits, dict):
            logits = logits['out']
        foreground = torch.sigmoid(logits[:, 0]) >= threshold
    if str(device).startswith('cuda') and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    scale = 1.0 / 255.0 if rgb.dtype.is_floating_point else 1.0
    red = torch.as_tensor((255, 0, 0), dtype=rgb.dtype, device=rgb.device) * scale
    gray = torch.as_tensor((128, 128, 128), dtype=rgb.dtype, device=rgb.device) * scale
    result = torch.where(foreground.to(rgb.device).unsqueeze(-1), red, gray)
    if single:
        result = result[0]
    color[..., :3].copy_(result)
    env._predmask_last_latency_s = time.perf_counter() - start
    return color


def load_segmenter(vision_dir, device):
    vision = Path(vision_dir)
    threshold = float(json.loads((vision / 'threshold.json').read_text(encoding='utf-8'))['threshold'])
    if threshold != 0.7:
        raise ValueError('Frozen ItemsHandover requires threshold 0.7')
    with (vision / 'best_model.ts').open('rb') as stream:
        model = torch.jit.load(stream, map_location=device)
    model.eval()
    return model, threshold
