"""V0 embedding grid (plan 2.3), GPU. Torch, timm and transformers are imported lazily (extra `embed`).

For each encoder, one pass per scale over the 8 dihedral views stores, per scale (chip 128 px; context
256 px re-cut from zoom-18 tiles), the CLS and mean-patch tokens of view 0 and their mean over all 8 views:
  <out>/<encoder>/<scale>_<pool>_<tta>.npy   scale in {chip, ctx}, pool in {cls, mean}, tta in {none, d8}
plus <out>/<encoder>/ids.npy. Any grid cell is then a concatenation of these arrays (see `features`).
Outputs are written under the HiME repo, never under the data root.
"""
import math
from pathlib import Path

import numpy as np

ENCODERS = {   # id: (library, model name)
    "dinov2_vits14": ("timm", "vit_small_patch14_dinov2.lvd142m"),
    "dinov2_vitb14": ("timm", "vit_base_patch14_dinov2.lvd142m"),
    "dinov3_vitb16": ("hf", "facebook/dinov3-vitb16-pretrain-lvd1689m"),
    "dinov3_vitl16_sat": ("hf", "facebook/dinov3-vitl16-pretrain-sat493m"),
}
IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
ZOOM, TILE = 18, 256


def chip_path(root, layer, bid):
    return Path(root) / "chips" / layer / str(int(bid) // 1000) / f"{int(bid)}.png"


def global_pixel(lat, lon, z=ZOOM):
    n = TILE * 2 ** z
    phi = math.radians(lat)
    return (lon + 180.0) / 360.0 * n, (1 - math.log(math.tan(phi) + 1 / math.cos(phi)) / math.pi) / 2 * n


def context_chip(root, layer, lat, lon, size=256, ext=None):
    """size x size RGB window centred on the building, stitched from zoom-18 tiles (missing tiles black)."""
    from PIL import Image
    gx, gy = global_pixel(lat, lon)
    x0, y0 = int(math.floor(gx)) - size // 2, int(math.floor(gy)) - size // 2
    out = np.zeros((size, size, 3), np.uint8)
    tdir = Path(root) / "tiles" / layer / str(ZOOM)
    for tx in range(x0 // TILE, (x0 + size - 1) // TILE + 1):
        for ty in range(y0 // TILE, (y0 + size - 1) // TILE + 1):
            cands = [tdir / str(tx) / f"{ty}.{e}" for e in ((ext,) if ext else ("jpg", "png", "webp"))]
            f = next((c for c in cands if c.exists()), None)
            if f is None:
                continue
            t = np.asarray(Image.open(f).convert("RGB"))
            ox, oy = tx * TILE - x0, ty * TILE - y0
            sx, sy = max(0, -ox), max(0, -oy)
            dx, dy = max(0, ox), max(0, oy)
            w, h = min(TILE - sx, size - dx), min(TILE - sy, size - dy)
            out[dy:dy + h, dx:dx + w] = t[sy:sy + h, sx:sx + w]
    return out


class _Dataset:
    def __init__(self, root, layer, ids, lat, lon, scale):
        self.root, self.layer, self.ids, self.lat, self.lon, self.scale = root, layer, ids, lat, lon, scale

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, k):
        from PIL import Image
        import torch
        if self.scale == "chip":
            p = chip_path(self.root, self.layer, self.ids[k])
            rgb = np.asarray(Image.open(p).convert("RGB")) if p.exists() else np.zeros((128, 128, 3), np.uint8)
        else:
            rgb = context_chip(self.root, self.layer, self.lat[k], self.lon[k])
        img = Image.fromarray(rgb).resize((224, 224), Image.BICUBIC)   # as the release: PIL bicubic to 224
        return torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1)


def load_model(enc, device):
    import torch
    lib, name = ENCODERS[enc]
    if lib == "timm":
        import timm
        m = timm.create_model(name, pretrained=True, num_classes=0, img_size=224)
        mean, std, prefix = IMAGENET[0], IMAGENET[1], m.num_prefix_tokens
        fwd = lambda x: m.forward_features(x)
    else:
        from transformers import AutoImageProcessor, AutoModel
        m = AutoModel.from_pretrained(name)
        proc = AutoImageProcessor.from_pretrained(name)
        mean, std = tuple(proc.image_mean), tuple(proc.image_std)
        prefix = 1 + int(getattr(m.config, "num_register_tokens", 0))
        fwd = lambda x: m(pixel_values=x).last_hidden_state
    m = m.eval().to(device)
    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1)
    return fwd, prefix, mean_t, std_t


def embed(enc, root, layer, ids, lat, lon, out, batch=128, workers=8, device="cuda", scales=("chip", "ctx"), d8=True):
    import torch
    out = Path(out) / enc; out.mkdir(parents=True, exist_ok=True)
    fwd, prefix, mean, std = load_model(enc, device)
    np.save(out / "ids.npy", np.asarray(ids))
    for scale in scales:
        done = [out / f"{scale}_{p}_{t}.npy" for p in ("cls", "mean") for t in (("none", "d8") if d8 else ("none",))]
        if all(f.exists() for f in done):
            continue
        dl = torch.utils.data.DataLoader(_Dataset(root, layer, ids, lat, lon, scale), batch_size=batch,
                                         num_workers=workers, pin_memory=True)
        acc = {k: [] for k in (("cls_none", "mean_none", "cls_d8", "mean_d8") if d8 else ("cls_none", "mean_none"))}
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            for x in dl:
                x = (x.to(device, non_blocking=True).float() / 255.0 - mean) / std
                cls_s, mean_s = 0, 0
                for k in range(8 if d8 else 1):                       # dihedral group D4
                    v = torch.rot90(x, k % 4, dims=(2, 3))
                    v = torch.flip(v, dims=(3,)) if k >= 4 else v
                    tok = fwd(v).float()
                    c, mp = tok[:, 0], tok[:, prefix:].mean(1)
                    if k == 0:
                        acc["cls_none"].append(c.cpu().numpy()); acc["mean_none"].append(mp.cpu().numpy())
                    cls_s, mean_s = cls_s + c, mean_s + mp
                if d8:
                    acc["cls_d8"].append((cls_s / 8).cpu().numpy()); acc["mean_d8"].append((mean_s / 8).cpu().numpy())
        for k, v in acc.items():
            np.save(out / f"{scale}_{k}.npy", np.concatenate(v).astype(np.float32))


class ColumnConcat:
    """Column-wise concatenation of memory-mapped arrays, materialised only for the rows indexed."""

    def __init__(self, parts):
        self.parts = parts
        self.shape = (len(parts[0]), sum(p.shape[1] for p in parts))
        self.dtype = parts[0].dtype

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, k):
        return np.hstack([np.asarray(p[k]) for p in self.parts])


def features(emb_root, enc, pool="cls", tta="none", ctx=False):
    """One V0 grid cell: pool in {cls, cls+mean}; tta in {none, d8}; ctx adds the 256 px context scale.
    Returns (ids, E) with E row-indexable; the full grid is never loaded into memory."""
    d = Path(emb_root) / enc
    parts = [("chip", "cls")] + ([("chip", "mean")] if pool == "cls+mean" else [])
    if ctx:
        parts += [("ctx", p) for _, p in parts]
    return np.load(d / "ids.npy"), ColumnConcat([np.load(d / f"{s}_{p}_{tta}.npy", mmap_mode="r") for s, p in parts])
