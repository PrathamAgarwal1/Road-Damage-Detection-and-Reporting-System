"""
RoadSight vision model: road gate + condition classifier on one MobileCLIP2-S2 image encoder.

Trained/exported by training/train.py into models/roadsight_clip.pt. At runtime only the image
encoder is built (text embeddings are precomputed in the checkpoint), keeping memory low.
"""
import os

import torch
from PIL import Image
from torchvision import transforms

MODEL_NAME, PRETRAINED = 'MobileCLIP2-S2', 'dfndr2b'
CLASS_NAMES = ['good', 'poor', 'satisfactory', 'very_poor']  # order used by the probe
IMAGE_SIZE = 256

ROAD_PROMPTS = [
    'a photo of a road', 'a photo of a street', 'a photo of a highway', 'a dashcam photo of a road',
    'a photo of a pothole in a road', 'a photo of a damaged road with cracks', 'a photo of asphalt pavement',
    'a close-up photo of a cracked asphalt road surface', 'a photo of a dirt road with potholes and puddles',
]
NOT_ROAD_PROMPTS = [
    'a photo of planets in space', 'a photo of outer space and stars', 'a photo of the moon', 'an illustration',
    'a cartoon', 'a screenshot of a website or app', 'a photo of a document with text', 'a page of text',
    'a photo of a person', 'a selfie', 'a group of people', 'a photo of an animal', 'a photo of a cat',
    'a photo of a dog', 'a photo of food', 'a photo of a drink', 'a photo of an indoor room',
    'a photo of a floor inside a building', 'a photo of a wall', 'a photo of a brick wall', 'a photo of grass',
    'a photo of gravel', 'a photo of a beach', 'a photo of a forest', 'a photo of mountains', 'a photo of the sky',
    'a photo of a building', 'a photo of an object', 'a microscope image', 'a medical scan', 'a diagram or chart',
    'a logo', 'a photo of a car interior', 'a pattern or texture', 'a photo of a rocket', 'a photo of a clock',
    'a photo of coins', 'a photo of a field or farm',
]
CONDITION_PROMPTS = {
    'good': ['a photo of a smooth, well-maintained road in good condition',
             'a photo of a new asphalt road with clear lane markings', 'a photo of an undamaged paved road'],
    'satisfactory': ['a photo of a road with minor wear and a few small cracks', 'a photo of a slightly worn asphalt road',
                     'a photo of a road with faded markings and light surface wear'],
    'poor': ['a photo of a road with many cracks and patched areas', 'a photo of a deteriorated road surface with cracking',
             'a photo of a worn road with some small potholes'],
    'very_poor': ['a photo of a road with large potholes', 'a photo of a severely damaged, broken road',
                  'a photo of a road full of potholes and puddles'],
}


def build_transform():
    """Exactly the preprocessing of the pretrained MobileCLIP2-S2 checkpoint (no mean/std normalisation)."""
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
    ])


def _build_visual(model_name):
    import open_clip
    try:  # build only the image tower (avoids allocating the ~63M-parameter text encoder)
        from open_clip.model import _build_vision_tower
        cfg = open_clip.get_model_config(model_name)
        return _build_vision_tower(cfg['embed_dim'], cfg['vision_cfg'])
    except Exception:
        return open_clip.create_model(model_name, pretrained=None).visual


class RoadVision:
    def __init__(self, path, device=None):
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.visual = _build_visual(ckpt['model_name'])
        self.visual.load_state_dict(ckpt.pop('visual_state_dict'))  # fp16 values are cast into the fp32 params in place
        self.visual.to(self.device).eval()
        self.transform = build_transform()
        self.class_names = ckpt['class_names']
        self.gate_text = ckpt['gate_text'].float()
        self.n_road = ckpt['n_road_prompts']
        self.cond_text = ckpt['cond_text'].float()
        self.probe_w, self.probe_b = ckpt['probe_weight'].float(), ckpt['probe_bias'].float()
        self.scale = ckpt['logit_scale']
        self.alpha = ckpt['blend_alpha']
        self.gate_accept, self.gate_review = ckpt['gate_accept'], ckpt['gate_review']
        self.metrics = ckpt.get('metrics', {})

    @torch.no_grad()
    def embed(self, image):
        x = self.transform(image.convert('RGB')).unsqueeze(0).to(self.device)
        return torch.nn.functional.normalize(self.visual(x).float(), dim=-1).cpu()[0]

    @torch.no_grad()
    def analyze(self, image_or_path):
        """
        Returns {
          'road_probability': float,             # zero-shot P(photo shows a road)
          'road_status': 'road' | 'review' | 'not_road',
          'probabilities': {class: percent},     # condition distribution (sums to 100)
          'condition': str, 'confidence': float  # argmax class and its percent
        }
        """
        image = Image.open(image_or_path) if isinstance(image_or_path, (str, os.PathLike)) else image_or_path
        z = self.embed(image)
        p_road = float((self.scale * z @ self.gate_text.T).softmax(-1)[:self.n_road].sum())
        status = 'road' if p_road >= self.gate_accept else ('review' if p_road >= self.gate_review else 'not_road')
        probe = (10 * z @ self.probe_w.T + self.probe_b).log_softmax(-1)
        zero_shot = (self.scale * z @ self.cond_text.T).log_softmax(-1)
        probs = ((1 - self.alpha) * probe + self.alpha * zero_shot).softmax(-1)
        pct = {c: round(float(p) * 100, 2) for c, p in zip(self.class_names, probs)}
        best = max(pct, key=pct.get)
        return {'road_probability': round(p_road, 4), 'road_status': status,
                'probabilities': pct, 'condition': best, 'confidence': pct[best]}
