"""Write a response to a pictured passage and question using a local checkpoint."""
import argparse
from pathlib import Path
import torch
from ilm.visual_lm.causal_glyph_writer import CausalGlyphWriter, packed_page, write_strip


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("passage", type=Path)
    p.add_argument("question", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--model", type=Path, default=Path("artifacts/causal_glyph_writer_v1/matched"))
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    torch.set_num_threads(4)
    w = CausalGlyphWriter.restore(args.model, args.device)
    start, sep = w.markers.split(1)
    source, prompt = [w.encode(packed_page(path).to(args.device)) for path in (args.passage,args.question)]
    result, ended = w.generate([torch.cat([start,source,sep,prompt,sep])], w.alphabet())
    chosen = torch.tensor(result[0], device=args.device, dtype=torch.long)
    write_strip(w.packed[chosen-1].cpu().numpy(), args.output)
    print(f"{len(chosen)} returned glyphs, stopped={ended[0]}")


if __name__ == "__main__":
    main()
