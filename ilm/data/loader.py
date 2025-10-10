from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from ilm.utils.glyphs import make_rgb_token_image, save_rgb_image
from ilm.db.glyph_db import GlyphDB


@dataclass
class Record:
    lang: str
    token: str
    path: str


class ImageIndexDataset(Dataset):
    """
    Dataset that reads an index.tsv with columns: lang \t token \t path
    Loads RGB PNG images into float tensors in [0,1].
    """

    def __init__(self, index_path: str, image_size: Optional[int] = None,
                 auto_generate_missing: bool = False,
                 glyph_db_path: Optional[str] = None):
        self.records: List[Record] = []
        with open(index_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                p = row["path"]
                p_norm = os.path.normpath(p)
                base = os.path.normpath(os.path.dirname(index_path))
                # Resolve path robustly:
                # 1) if p exists as given, use it
                # 2) else if base/p exists, use that
                # 3) else keep base/p as target (for auto-generation)
                if os.path.isabs(p_norm) and os.path.exists(p_norm):
                    p_abs = p_norm
                elif os.path.exists(p_norm):
                    p_abs = p_norm
                else:
                    p_candidate = os.path.normpath(os.path.join(base, p_norm))
                    if os.path.exists(p_candidate):
                        p_abs = p_candidate
                    else:
                        p_abs = p_candidate
                self.records.append(Record(lang=row["lang"], token=row["token"], path=p_abs))
        self.image_size = image_size
        self.auto_generate_missing = auto_generate_missing
        self.glyph_db: Optional[GlyphDB] = GlyphDB(glyph_db_path) if glyph_db_path else None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        # Prefer DB if available
        if self.glyph_db is not None:
            path = self.glyph_db.ensure_glyph(rec.lang, rec.token, size=self.image_size or 128)
        else:
            path = rec.path
            if not os.path.exists(path) and self.auto_generate_missing:
                try:
                    rgb = make_rgb_token_image(rec.lang, rec.token, size=self.image_size or 128)
                    save_rgb_image(path, rgb)
                except Exception as e:
                    raise FileNotFoundError(f"Failed to generate missing glyph for {rec.lang}:{rec.token} at {path}: {e}")
        img = Image.open(path).convert("RGB")
        if self.image_size is not None:
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        # CHW tensor
        t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        return {"image": t, "lang": rec.lang, "token": rec.token, "path": rec.path}


def make_dataloader(index_path: str, batch_size: int = 256, shuffle: bool = True,
                    num_workers: int = 4, image_size: Optional[int] = None,
                    auto_generate_missing: bool = False,
                    glyph_db_path: Optional[str] = None) -> DataLoader:
    ds = ImageIndexDataset(index_path=index_path, image_size=image_size,
                           auto_generate_missing=auto_generate_missing,
                           glyph_db_path=glyph_db_path)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=True)
