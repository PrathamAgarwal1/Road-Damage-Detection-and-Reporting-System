"""
Build the NON-road evaluation set used to verify the road gate (training/negatives/).

Real photos bundled with scikit-image (space, people, animals, objects, text, textures) and
the ultralytics sample 'zidane.jpg', plus rendered images of typical junk uploads: a solar-system
illustration, a document, a UI screenshot and plain floors.

Usage:  python training/make_negatives.py      (needs: pip install scikit-image)
"""
import os
import random
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'negatives')


def save(img, name):
    img.convert('RGB').save(os.path.join(OUT, name + '.jpg'), quality=92)


def solar_system(seed=0, w=1200, h=700):
    rnd = random.Random(seed)
    img = Image.new('RGB', (w, h), (4, 4, 14))
    d = ImageDraw.Draw(img)
    for _ in range(900):
        x, y, b = rnd.randrange(w), rnd.randrange(h), rnd.randrange(80, 255)
        d.point((x, y), fill=(b, b, b))
    sun_r = 260
    for r in range(sun_r, 0, -4):
        t = r / sun_r
        d.ellipse([-sun_r // 2 - r, h // 2 - r, -sun_r // 2 + r, h // 2 + r], fill=(255, int(120 + 120 * (1 - t)), int(20 * (1 - t))))
    x = 230
    palette = [(170, 160, 150), (220, 180, 120), (70, 120, 220), (200, 90, 60), (210, 170, 130), (220, 200, 150), (150, 210, 230), (60, 90, 200)]
    for i, col in enumerate(palette):
        r = rnd.choice([14, 20, 26, 55, 70, 45, 34, 32]) if i > 3 else rnd.choice([10, 16, 18, 13])
        cx, cy = x + r, h // 2 + rnd.randint(-40, 40)
        for k in range(r, 0, -1):  # shaded sphere
            f = 0.45 + 0.55 * (1 - k / r)
            d.ellipse([cx - k - (r - k) // 3, cy - k - (r - k) // 3, cx + k - (r - k) // 3, cy + k - (r - k) // 3],
                      fill=tuple(int(c * f) for c in col))
        if i == 5:
            d.ellipse([cx - r * 1.9, cy - r * 0.45, cx + r * 1.9, cy + r * 0.45], outline=(230, 210, 170), width=5)
        x += 2 * r + rnd.randint(35, 70)
    return img.filter(ImageFilter.GaussianBlur(0.6))


def document():
    doc = Image.new('RGB', (900, 1200), 'white')
    d = ImageDraw.Draw(doc)
    f = ImageFont.load_default(size=26)
    for i in range(30):
        d.text((60, 60 + i * 36), f'Q{i + 1}. Explain the working of a transistor amplifier with a diagram.', fill='black', font=f)
    return doc


def screenshot():
    img = Image.new('RGB', (1280, 800), (243, 244, 246))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 1280, 60], fill=(37, 99, 235))
    d.rectangle([0, 60, 240, 800], fill=(255, 255, 255))
    f = ImageFont.load_default(size=20)
    for i in range(12):
        d.text((24, 90 + i * 50), f'Menu item {i + 1}', fill=(55, 65, 81), font=f)
    for r in range(3):
        for c in range(3):
            x, y = 280 + c * 330, 100 + r * 230
            d.rounded_rectangle([x, y, x + 300, y + 200], 14, fill='white', outline=(229, 231, 235))
            d.text((x + 20, y + 20), f'Card {r * 3 + c + 1}', fill=(17, 24, 39), font=f)
    return img


def main():
    os.makedirs(OUT, exist_ok=True)
    import skimage.data as sk
    for name in ['astronaut', 'camera', 'coffee', 'chelsea', 'moon', 'page', 'text', 'rocket', 'hubble_deep_field',
                 'brick', 'grass', 'gravel', 'coins', 'horse', 'clock', 'cell', 'immunohistochemistry', 'retina',
                 'colorwheel', 'logo', 'cat', 'checkerboard']:
        arr = getattr(sk, name)()
        save(Image.fromarray(arr if arr.dtype == np.uint8 else (arr * 255).astype(np.uint8)), f'sk_{name}')
    try:
        import ultralytics
        shutil.copy(os.path.join(os.path.dirname(ultralytics.__file__), 'assets', 'zidane.jpg'), os.path.join(OUT, 'people_zidane.jpg'))
    except Exception:
        pass
    for s in range(3):
        save(solar_system(s), f'planets_{s}')
    save(document(), 'document')
    save(screenshot(), 'ui_screenshot')
    rng = np.random.default_rng(0)
    save(Image.fromarray((rng.normal(128, 18, (800, 1000, 1)).clip(0, 255).repeat(3, 2)).astype('uint8')), 'grey_floor')
    tile = Image.new('RGB', (1000, 800), (200, 190, 175))
    d = ImageDraw.Draw(tile)
    for x in range(0, 1000, 125):
        d.line([(x, 0), (x, 800)], fill=(150, 145, 135), width=4)
    for y in range(0, 800, 125):
        d.line([(0, y), (1000, y)], fill=(150, 145, 135), width=4)
    save(tile, 'tiled_floor')
    print(f'{len(os.listdir(OUT))} negative images in {OUT}')


if __name__ == '__main__':
    main()
