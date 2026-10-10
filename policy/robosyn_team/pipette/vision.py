"""Four-class SegFormer-B0 and one shared predicted RGB overlay path."""
# Adapted from project-018/work/vision.py; the supplied folder replaces ROOT.
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation
from .pipette_contract import LABELS, overlay


def image_tensor(images, device):
    array = np.stack(images)
    if array.dtype != np.uint8 or array.shape[1:] != (480, 640, 3):
        raise ValueError('Expected 480x640 RGB uint8 images')
    x = torch.from_numpy(array).to(device).permute(0, 3, 1, 2).float()/255
    return (x-x.new_tensor([.485,.456,.406])[None,:,None,None])/x.new_tensor([.229,.224,.225])[None,:,None,None]


class Predictor:
    def __init__(self, folder, device='cuda'):
        self.folder = Path(folder)
        self.device = device
        self.model = SegformerForSemanticSegmentation.from_pretrained(str(self.folder), local_files_only=True).to(device).eval()
        cfg = json.loads((self.folder/'renderer.json').read_text())
        if cfg != {'classes':list(LABELS), 'alpha':.25, 'confidence':.6}:
            raise ValueError('Renderer identity mismatch')

    @torch.inference_mode()
    def predict(self, images):
        logits = self.model(pixel_values=image_tensor(images,self.device)).logits
        probabilities = F.interpolate(logits.float(),(480,640),mode='bilinear',align_corners=False).softmax(1)
        confidence, labels = probabilities.max(1)
        return labels.cpu().numpy().astype(np.uint8), confidence.cpu().numpy()

    def enhance(self, images):
        labels, confidence = self.predict(images)
        return [overlay(rgb,label,conf) for rgb,label,conf in zip(images,labels,confidence)]
