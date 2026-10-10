"""Identical learned mask processing with explicit local inference paths."""
import numpy as np


class Predictor:
    def __init__(self, folder, kind, device='cuda'):
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
        self.kind = kind
        self.device = device
        self.model = SegformerForSemanticSegmentation.from_pretrained(folder, local_files_only=True).to(device).eval()
        self.processor = SegformerImageProcessor.from_pretrained(folder, local_files_only=True)
        if self.model.config.num_labels != len(kind.CLASSES):
            raise ValueError('Expected eight selected semantic classes')

    def predict(self, images):
        import torch
        if len(images) != 3 or any(im.dtype != np.uint8 or im.shape != (288,384,3) for im in images):
            raise ValueError('Expected three 384x288 RGB images')
        with torch.inference_mode():
            pixels = self.processor(images=images, return_tensors='pt')['pixel_values'].to(self.device)
            logits = self.model(pixel_values=pixels).logits
            return torch.nn.functional.interpolate(logits, size=(288,384), mode='bilinear',
                                                   align_corners=False).argmax(1).cpu().numpy()

    def enhance(self, images):
        return [self.kind.enhance(rgb, mask) for rgb, mask in zip(images, self.predict(images))]
