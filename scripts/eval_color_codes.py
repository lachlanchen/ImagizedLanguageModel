#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import base64
import io

try:
    from ilm.data.loader import ImageIndexDataset
    from ilm.encoders.glyph_cnn import GlyphCNN
    from ilm.code.product import ProductCode
except ModuleNotFoundError:
    import sys
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from ilm.data.loader import ImageIndexDataset
    from ilm.encoders.glyph_cnn import GlyphCNN
    from ilm.code.product import ProductCode


def load_checkpoint(path: str, device: str) -> Tuple[Dict, Dict]:
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt.get("cfg", {})
    return ckpt, cfg


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def eval_metrics(index_path: str, ckpt_path: str, out_root: str,
                 batch_size: int = 256, image_size: int | None = 128,
                 device: str = "cuda", auto_generate_missing: bool = False,
                 pairs_path: str | None = None):
    # Output directory with datetime suffix
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(out_root) / ts
    ensure_dir(out_dir)

    # Load checkpoint and config
    ckpt, cfg = load_checkpoint(ckpt_path, device)
    d_glyph = cfg.get("model", {}).get("d_glyph", 128)
    d_code = cfg.get("model", {}).get("d_code", 128)
    K = cfg.get("model", {}).get("K", 32)
    C = cfg.get("model", {}).get("C", 3)

    # Build dataset/loader
    ds = ImageIndexDataset(index_path=index_path, image_size=image_size,
                           auto_generate_missing=auto_generate_missing)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Instantiate models
    glyph_cnn = GlyphCNN(d=d_glyph, in_channels=3).to(device)
    code = ProductCode(d_in=d_glyph, d=d_code, K=K, C=C, tau=0.2, straight_through=False).to(device)
    glyph_cnn.load_state_dict(ckpt["glyph_cnn"])  # type: ignore[index]
    code.load_state_dict(ckpt["code"])            # type: ignore[index]
    glyph_cnn.eval()
    code.eval()

    # Accumulators
    N = 0
    sum_y = [torch.zeros(K, device=device) for _ in range(C)]
    sum_outer = [[torch.zeros(K, K, device=device) for _ in range(C)] for __ in range(C)]
    embeds = []        # code embeddings
    glyph_feats = []   # glyph cnn outputs
    langs: List[str] = []
    tokens: List[str] = []
    paths: List[str] = []

    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            g = glyph_cnn(x)              # (B×d_glyph)
            out = code(g, tau=0.2)
            y = out["y"]                  # (B×C×K)
            e = out["embed"]             # (B×d_code)

            B = x.size(0)
            N += B

            # Sums per channel
            for c in range(C):
                yc = y[:, c, :]  # (B×K)
                sum_y[c] += yc.sum(dim=0)
            # Cross outer products for independence
            for a in range(C):
                Ya = y[:, a, :]
                for b in range(a + 1, C):
                    Yb = y[:, b, :]
                    sum_outer[a][b] += Ya.t() @ Yb

            embeds.append(e.detach().cpu())
            glyph_feats.append(g.detach().cpu())
            langs.extend(batch["lang"])  # type: ignore[arg-type]
            tokens.extend(batch["token"])  # type: ignore[arg-type]
            paths.extend(batch["path"])  # type: ignore[arg-type]

    embeds = torch.cat(embeds, dim=0).numpy().astype(np.float16)
    glyph_feats = torch.cat(glyph_feats, dim=0).numpy().astype(np.float16)

    # Usage and entropy
    usage = []
    entropies = []
    for c in range(C):
        p = (sum_y[c] / N).clamp_min(1e-9)
        usage.append(p.detach().cpu().numpy())
        H = - (p * p.log()).sum().item()
        entropies.append(H)

    # Independence metric
    indep_vals = []
    means = [sum_y[c] / N for c in range(C)]
    for a in range(C):
        for b in range(a + 1, C):
            S = sum_outer[a][b] / N
            cov = S - torch.ger(means[a], means[b])
            indep_vals.append(float((cov.pow(2).sum()).item()))
    indep_mean = float(np.mean(indep_vals)) if indep_vals else 0.0

    # Save arrays and ids
    np.save(out_dir / "codes.npy", embeds)
    np.save(out_dir / "glyphs.npy", glyph_feats)
    with open(out_dir / "ids.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["idx", "lang", "token", "path"])  # header
        for i, (la, tok, pa) in enumerate(zip(langs, tokens, paths)):
            w.writerow([i, la, tok, pa])

    # Optional: cross-lingual retrieval metrics
    retrieval = None
    if pairs_path and os.path.exists(pairs_path):
        retrieval = compute_retrieval(out_dir, pairs_path)

    # Summarize
    summary = {
        "dataset_size": N,
        "channels": C,
        "K": K,
        "entropy": entropies,
        "independence_mean": indep_mean,
        "retrieval": retrieval,
        "checkpoint": ckpt_path,
        "index": index_path,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Save usage CSV
    with open(out_dir / "usage.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["channel", "bin", "prob"])
        for c in range(C):
            for k in range(K):
                w.writerow([c, k, float(usage[c][k])])

    # Save a few NN examples
    save_nn_examples(out_dir)
    # Write HTML report (best-effort)
    try:
        write_html_report(out_dir, usage, entropies, indep_mean)
    except Exception as e:
        with open(out_dir / "report_error.txt", "w", encoding="utf-8") as f:
            f.write(str(e))

    print(json.dumps({"metrics_dir": str(out_dir)}))


def compute_retrieval(out_dir: Path, pairs_path: str) -> Dict[str, float]:
    codes = np.load(out_dir / "codes.npy").astype(np.float32)
    ids = []
    with open(out_dir / "ids.tsv", "r", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            ids.append(row)
    # Indices by token/lang
    idx_by_tok = {}
    for row in ids:
        idx_by_tok.setdefault((row["lang"], row["token"]), int(row["idx"]))
    # Read pairs
    pairs = []
    with open(pairs_path, "r", encoding="utf-8") as f:
        pr = csv.DictReader(f, delimiter="\t")
        for row in pr:
            if "en" in row and "zh" in row:
                pairs.append((row["en"], row["zh"]))
    # Build matrices
    en_idx = [idx_by_tok.get(("en", t)) for t, _ in pairs]
    zh_idx = [idx_by_tok.get(("zh", t)) for _, t in pairs]
    # Filter missing
    keep = [i for i, (e, z) in enumerate(zip(en_idx, zh_idx)) if e is not None and z is not None]
    en_idx = [en_idx[i] for i in keep]
    zh_idx = [zh_idx[i] for i in keep]
    if len(en_idx) == 0:
        return {"R@1": 0.0, "R@5": 0.0, "R@10": 0.0, "pairs_used": 0}
    E = codes[en_idx]
    Z = codes
    # restrict Z to zh only
    zh_rows = [int(row["idx"]) for row in ids if row["lang"] == "zh"]
    Z = codes[zh_rows]
    # normalize
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    sims = E @ Z.T
    # compute ranks
    recall = {1: 0, 5: 0, 10: 0}
    for qi, eidx in enumerate(en_idx):
        # the correct zh index among zh_rows
        true_zh_global = zh_idx[qi]
        try:
            true_pos = zh_rows.index(true_zh_global)
        except ValueError:
            continue
        order = np.argsort(-sims[qi])
        rank = int(np.where(order == true_pos)[0]) + 1
        for k in recall.keys():
            if rank <= k:
                recall[k] += 1
    total = len(en_idx)
    res = {f"R@{k}": float(recall[k] / total) for k in recall}
    res["pairs_used"] = int(total)
    return res


def save_nn_examples(out_dir: Path, n_samples: int = 10, k: int = 10):
    codes = np.load(out_dir / "codes.npy").astype(np.float32)
    ids = []
    with open(out_dir / "ids.tsv", "r", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            ids.append(row)
    # normalize
    codes = codes / (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-9)
    n = len(ids)
    rng = np.random.default_rng(0)
    samples = rng.choice(n, size=min(n_samples, n), replace=False)
    with open(out_dir / "nn_examples.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["query_idx", "query_lang", "query_token", "nn_rank", "nn_idx", "nn_lang", "nn_token", "sim"])
        for qi in samples:
            sims = codes[qi:qi+1] @ codes.T
            order = np.argsort(-sims[0])
            for r, j in enumerate(order[:k]):
                w.writerow([
                    qi, ids[qi]["lang"], ids[qi]["token"],
                    r + 1, j, ids[j]["lang"], ids[j]["token"], float(sims[0, j])
                ])


def _encode_fig_png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _plot_usage_images(usage: List[np.ndarray]) -> List[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []
    imgs = []
    for c, p in enumerate(usage):
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.bar(np.arange(len(p)), p, color="#4e79a7")
        ax.set_title(f"Channel {c} usage")
        ax.set_xlabel("bin (K)")
        ax.set_ylabel("probability")
        ax.set_ylim(0, max(0.05, float(p.max()) * 1.1))
        data_uri = _encode_fig_png(fig)
        imgs.append(data_uri)
        plt.close(fig)
    return imgs


def _pca_scatter_image(codes: np.ndarray, langs: List[str], max_points: int = 2000) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    n = codes.shape[0]
    idx = np.random.default_rng(0).choice(n, size=min(max_points, n), replace=False)
    X = codes[idx].astype(np.float32)
    L = [langs[i] for i in idx]
    Xc = X - X.mean(axis=0, keepdims=True)
    # PCA via SVD
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    coords = Xc @ Vt[:2].T
    colors = ["#4e79a7" if l == "en" else ("#e15759" if l == "zh" else "#76b7b2") for l in L]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=6, alpha=0.6, linewidths=0)
    ax.set_title("Codes PCA (sample)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    data_uri = _encode_fig_png(fig)
    plt.close(fig)
    return data_uri


def write_html_report(out_dir: Path, usage: List[np.ndarray], entropies: List[float], indep_mean: float) -> None:
    # Load ids and codes
    codes = np.load(out_dir / "codes.npy")
    langs = []
    tokens = []
    with open(out_dir / "ids.tsv", "r", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            langs.append(row["lang"])  # type: ignore[index]
            tokens.append(row["token"])  # type: ignore[index]

    usage_imgs = _plot_usage_images(usage)
    pca_img = _pca_scatter_image(codes, langs)

    # Load top-NN table (first 50 rows)
    nn_rows = []
    nn_path = out_dir / "nn_examples.tsv"
    if nn_path.exists():
        with open(nn_path, "r", encoding="utf-8") as f:
            r = csv.DictReader(f, delimiter="\t")
            for i, row in enumerate(r):
                if i >= 50:
                    break
                nn_rows.append(row)

    # Compose HTML
    html = [
        "<html><head><meta charset='utf-8'><title>Color Codes Metrics</title>",
        "<style>body{font-family:sans-serif;max-width:1000px;margin:20px auto;} table{border-collapse:collapse;width:100%;} th,td{border:1px solid #ccc;padding:4px 6px;} .sec{margin:20px 0;} .imgs{display:flex;gap:12px;flex-wrap:wrap;} .imgbox{border:1px solid #ddd;padding:6px}</style>",
        "</head><body>",
        f"<h1>Product Color Codes Report</h1>",
        f"<div class='sec'><h2>Summary</h2>",
        f"<p>Entropy per channel: {', '.join(f'{h:.3f}' for h in entropies)}<br/>Independence (mean Frob^2): {indep_mean:.6f}</p>",
        f"<p>Artifacts: <a href='summary.json'>summary.json</a>, <a href='usage.csv'>usage.csv</a>, <a href='ids.tsv'>ids.tsv</a>, <a href='codes.npy'>codes.npy</a>, <a href='glyphs.npy'>glyphs.npy</a></p>",
        "</div>",
    ]

    # Usage images
    html.append("<div class='sec'><h2>Channel Usage</h2><div class='imgs'>")
    if usage_imgs:
        for uri in usage_imgs:
            html.append(f"<div class='imgbox'><img src='{uri}'/></div>")
    else:
        html.append("<p>(matplotlib unavailable; see usage.csv)</p>")
    html.append("</div></div>")

    # PCA image
    html.append("<div class='sec'><h2>PCA of Codes (sample)</h2>")
    if pca_img:
        html.append(f"<div class='imgbox'><img src='{pca_img}'/></div>")
    else:
        html.append("<p>(matplotlib unavailable)</p>")
    html.append("</div>")

    # NN examples
    html.append("<div class='sec'><h2>Nearest Neighbors (sample)</h2>")
    if nn_rows:
        html.append("<table><thead><tr><th>q_idx</th><th>q_lang</th><th>q_token</th><th>rank</th><th>nn_idx</th><th>nn_lang</th><th>nn_token</th><th>sim</th></tr></thead><tbody>")
        for r in nn_rows:
            html.append("<tr>" + "".join(
                f"<td>{r[k]}</td>" for k in ["query_idx","query_lang","query_token","nn_rank","nn_idx","nn_lang","nn_token","sim"]
            ) + "</tr>")
        html.append("</tbody></table>")
    else:
        html.append("<p>No NN examples available.</p>")
    html.append("</div>")

    html.append("</body></html>")

    with open(out_dir / "report.html", "w", encoding="utf-8") as f:
        f.write("\n".join(html))


def main():
    ap = argparse.ArgumentParser(description="Evaluate product color codes and generate metrics with timestamped folder")
    ap.add_argument("--index", default="data/processed/images_common_freq/index.tsv")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-root", default="artifacts/metrics")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--auto-generate-missing", action="store_true")
    ap.add_argument("--pairs", default=None, help="Optional bilingual pairs TSV with columns: en\tzh")
    args = ap.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    eval_metrics(
        index_path=args.index,
        ckpt_path=args.checkpoint,
        out_root=args.out_root,
        batch_size=args.batch_size,
        image_size=args.image_size,
        device=device,
        auto_generate_missing=args.auto_generate_missing,
        pairs_path=args.pairs,
    )


if __name__ == "__main__":
    main()
