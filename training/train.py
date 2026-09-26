"""
Train and export the RoadSight vision model.

Pipeline (one small CLIP image encoder, MobileCLIP2-S2, does everything):
  1. Road gate    — zero-shot: P(road) from "road" vs. ~40 "not a road" prompts (planets, documents,
                    people, food, rooms, textures …). No training data needed, so it generalises.
  2. Condition    — linear probe (multinomial logistic regression) on the frozen image embedding,
                    trained on road_damage_dataset, blended with zero-shot condition prompts.
                    The blend counters the dataset's camera bias (each class comes from a different
                    source; e.g. every 'very_poor' image is a web photo), which otherwise makes any
                    web-style photo look 'very poor'.

Data handling: near-duplicate images (perceptual hash) are kept in the same split, and duplicate
groups with conflicting labels are dropped. Split 70/15/15 (seed 42) is saved to training/split.json.

Usage:
    pip install open_clip_torch scikit-image
    python training/make_negatives.py     # non-road evaluation images
    python training/train.py

Outputs: models/roadsight_clip.pt (weights + config), training/metrics.json, training/split.json
"""
import collections
import glob
import json
import os
import random
import sys

import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault('MONGO_URI', 'mongodb://localhost:27017/roadsight_train')  # services import only; no DB access
import services  # noqa: E402  (perceptual hash)
from vision import (CLASS_NAMES, ROAD_PROMPTS, NOT_ROAD_PROMPTS, CONDITION_PROMPTS, MODEL_NAME, PRETRAINED,  # noqa: E402
                    build_transform)

DATASET = os.path.join(ROOT, 'road_damage_dataset')
OUT_MODEL = os.path.join(ROOT, 'models', 'roadsight_clip.pt')
HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 42
LOGIT_SCALE = 100.0
MAX_VAL_DROP = 0.015          # blend weight: largest alpha that costs <= 1.5 pts of validation accuracy
GATE_ACCEPT, GATE_REVIEW = 0.5, 0.15

# Real photos from outside the dataset, labelled by hand, used only for reporting
OOD = {
    'static/uploads/images.jpg': 'good',
    'static/uploads/road_20251104_233548_g.jpeg': 'good',
    'static/uploads/road_20251104_230107_e.jpg': 'satisfactory',
    'static/uploads/road_20251104_225823_w.jpg': 'very_poor',
    'static/uploads/Screenshot 2025-08-28 162100.png': 'very_poor',
    'static/uploads/road_damaged12.jpg': 'very_poor',
    'static/uploads/road_20251022_173041_A_3200_0.jpg': 'very_poor',
}


def make_split():
    items = [(c, os.path.join(DATASET, c, f)) for c in CLASS_NAMES
             for f in sorted(os.listdir(os.path.join(DATASET, c))) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    groups = collections.defaultdict(list)
    for c, p in items:
        groups[services.image_dhash(p)].append((c, p))
    clean, conflicts = [], []
    for g in groups.values():
        (conflicts if len({c for c, _ in g}) > 1 else clean).append(g)
    rnd = random.Random(SEED)
    split = {'train': [], 'val': [], 'test': []}
    for c in CLASS_NAMES:
        gs = [g for g in clean if g[0][0] == c]
        rnd.shuffle(gs)
        a, b = int(len(gs) * .70), int(len(gs) * .85)
        for name, part in (('train', gs[:a]), ('val', gs[a:b]), ('test', gs[b:])):
            split[name] += [x for g in part for x in g]
    rel = lambda p: os.path.relpath(p, DATASET).replace('\\', '/')
    json.dump({'seed': SEED, 'dropped_conflicting': [rel(p) for g in conflicts for _, p in g],
               **{k: [rel(p) for _, p in v] for k, v in split.items()}},
              open(os.path.join(HERE, 'split.json'), 'w'), indent=1)
    return split, sum(len(g) for g in conflicts)


def main():
    import open_clip
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(SEED)
    split, dropped = make_split()
    print('split sizes', {k: len(v) for k, v in split.items()}, '| dropped conflicting-label duplicates:', dropped)

    model, _, _ = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    model = model.to(dev).eval()
    tok = open_clip.get_tokenizer(MODEL_NAME)
    tf = build_transform()

    @torch.no_grad()
    def embed(paths):
        out = []
        for i in range(0, len(paths), 64):
            x = torch.stack([tf(Image.open(p).convert('RGB')) for p in paths[i:i + 64]]).to(dev)
            out.append(torch.nn.functional.normalize(model.encode_image(x).float(), dim=-1).cpu())
        return torch.cat(out)

    @torch.no_grad()
    def text(prompts):
        return torch.nn.functional.normalize(model.encode_text(tok(prompts).to(dev)).float(), dim=-1).cpu()

    E = {k: embed([p for _, p in v]) for k, v in split.items()}
    y = {k: torch.tensor([CLASS_NAMES.index(c) for c, _ in v]) for k, v in split.items()}
    gate_text = text(ROAD_PROMPTS + NOT_ROAD_PROMPTS)
    cond_text = torch.nn.functional.normalize(torch.stack([text(CONDITION_PROMPTS[c]).mean(0) for c in CLASS_NAMES]), dim=-1)

    def p_road(Z):
        return (LOGIT_SCALE * Z @ gate_text.T).softmax(-1)[:, :len(ROAD_PROMPTS)].sum(-1)

    def zs_logp(Z):
        return (LOGIT_SCALE * Z @ cond_text.T).log_softmax(-1)

    # --- linear probe with class-balanced loss; weight decay picked on validation ---
    def train_probe(wd):
        torch.manual_seed(SEED)
        W = torch.nn.Linear(E['train'].shape[1], len(CLASS_NAMES))
        opt = torch.optim.LBFGS(W.parameters(), max_iter=500, line_search_fn='strong_wolfe')
        freq = torch.bincount(y['train'], minlength=len(CLASS_NAMES)).float()
        cw = (1 / freq) / (1 / freq).mean()

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(W(E['train'] * 10), y['train'], weight=cw) + wd * W.weight.pow(2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        return W

    acc = lambda lp, yy: float((lp.argmax(1) == yy).float().mean())
    with torch.no_grad():
        wd = max((1e-4, 1e-3, 1e-2), key=lambda w: acc(train_probe(w)(E['val'] * 10), y['val']))
        W = train_probe(wd)
        probe = lambda Z: W(Z * 10).log_softmax(-1)
        blend = lambda Z, a: (1 - a) * probe(Z) + a * zs_logp(Z)
        best_val = acc(probe(E['val']), y['val'])
        alpha = max(a for a in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5) if acc(blend(E['val'], a), y['val']) >= best_val - MAX_VAL_DROP)

        pred = blend(E['test'], alpha).argmax(1)
        cm = torch.zeros(4, 4, dtype=torch.int)
        for t, p in zip(y['test'], pred):
            cm[t, p] += 1
        negatives = sorted(glob.glob(os.path.join(HERE, 'negatives', '*.jpg')))
        En = embed(negatives)
        ood_paths = [os.path.join(ROOT, p) for p in OOD if os.path.exists(os.path.join(ROOT, p))]
        Eo = embed(ood_paths)
        yo = torch.tensor([CLASS_NAMES.index(OOD[os.path.relpath(p, ROOT).replace('\\', '/')]) for p in ood_paths])
        pr_all = p_road(torch.cat(list(E.values())))
        pr_neg, pr_ood = p_road(En), p_road(Eo)

        metrics = {
            'model': f'{MODEL_NAME}/{PRETRAINED}', 'probe_weight_decay': wd, 'blend_alpha': alpha,
            'split_sizes': {k: len(v) for k, v in split.items()}, 'dropped_conflicting_duplicates': dropped,
            'val_accuracy': round(acc(blend(E['val'], alpha), y['val']), 4),
            'test_accuracy': round(acc(blend(E['test'], alpha), y['test']), 4),
            'test_accuracy_probe_only': round(acc(probe(E['test']), y['test']), 4),
            'test_confusion': {'classes': CLASS_NAMES, 'rows_true_cols_pred': cm.tolist()},
            'test_recall': {c: round(cm[i, i].item() / max(cm[i].sum().item(), 1), 4) for i, c in enumerate(CLASS_NAMES)},
            'out_of_dataset_accuracy': f'{int((blend(Eo, alpha).argmax(1) == yo).sum())}/{len(yo)}',
            'gate': {
                'accept_threshold': GATE_ACCEPT, 'review_threshold': GATE_REVIEW,
                'dataset_roads_accepted': round(float((pr_all >= GATE_ACCEPT).float().mean()), 4),
                'dataset_roads_rejected': int((pr_all < GATE_REVIEW).sum()),
                'non_road_images_passed': f'{int((pr_neg >= GATE_REVIEW).sum())}/{len(negatives)}',
                'max_non_road_score': round(float(pr_neg.max()), 4),
                'out_of_dataset_roads_accepted': f'{int((pr_ood >= GATE_REVIEW).sum())}/{len(ood_paths)}',
            },
        }

    os.makedirs(os.path.dirname(OUT_MODEL), exist_ok=True)
    torch.save({
        'model_name': MODEL_NAME, 'pretrained': PRETRAINED, 'class_names': CLASS_NAMES,
        'visual_state_dict': {k: v.half().cpu() for k, v in model.visual.state_dict().items()},
        'gate_text': gate_text.half(), 'n_road_prompts': len(ROAD_PROMPTS), 'cond_text': cond_text.half(),
        'probe_weight': W.weight.detach().clone(), 'probe_bias': W.bias.detach().clone(),
        'logit_scale': LOGIT_SCALE, 'blend_alpha': alpha,
        'gate_accept': GATE_ACCEPT, 'gate_review': GATE_REVIEW, 'metrics': metrics,
    }, OUT_MODEL)
    json.dump(metrics, open(os.path.join(HERE, 'metrics.json'), 'w'), indent=2)
    print(json.dumps(metrics, indent=2))
    print(f'Saved {OUT_MODEL} ({os.path.getsize(OUT_MODEL) / 1e6:.0f} MB)')


if __name__ == '__main__':
    main()
