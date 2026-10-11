"""RGB-only SegFormer prediction; deployment never reads simulator labels."""
import json
from pathlib import Path
import numpy as np


def validate_images(images):
    if len(images) != 3 or any(im.dtype != np.uint8 or im.ndim != 3 or im.shape[-1] != 3 for im in images):
        raise ValueError('Expected three RGB uint8 images')
    if any(im.shape != images[0].shape for im in images):
        raise ValueError('Camera image dimensions differ')


def image_tensor(images, device):
    import torch
    array = np.stack(images)
    x = torch.from_numpy(array).to(device).permute(0,3,1,2).float()/255
    return (x-x.new_tensor([.485,.456,.406])[None,:,None,None])/x.new_tensor([.229,.224,.225])[None,:,None,None]


class Predictor:
    def __init__(self, folder, device='cuda'):
        from transformers import SegformerForSemanticSegmentation
        from .bell_contract import LABELS
        self.device = device
        self.folder = Path(folder)
        renderer = json.loads((self.folder/'renderer.json').read_text())
        if renderer != {'classes':list(LABELS),'foreground':[255,0,0],'background':[128,128,128]}:
            raise ValueError('Learned-mask renderer identity changed')
        self.model = SegformerForSemanticSegmentation.from_pretrained(str(self.folder),local_files_only=True).to(device).eval()
        if self.model.config.num_labels != 3:
            raise ValueError('Expected background, robot, bell predictor')

    def predict(self, images):
        import torch
        import torch.nn.functional as F
        validate_images(images)
        with torch.inference_mode():
            logits = self.model(pixel_values=image_tensor(images,self.device)).logits
            labels = F.interpolate(logits.float(),images[0].shape[:2],mode='bilinear',align_corners=False).argmax(1)
        return labels.cpu().numpy().astype(np.uint8)

    def enhance(self, images):
        from .bell_contract import render_redgray
        return [render_redgray(labels) for labels in self.predict(images)]
