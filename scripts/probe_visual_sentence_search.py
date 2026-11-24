"""Frozen fourth-cohort diagnostic of complete sequence probability and search."""
from collections import Counter, defaultdict
from hashlib import sha256
import numpy as np
import torch
from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.stopped_search import search, reconstruct
from scripts.prepare_sentence_context import FOURTH, ROOT, Ink, read, dump, sha
from scripts.evaluate_visual_discourse import unpack
from scripts.write_fact_ranking import RankedAnswer
from scripts.train_region_writer import pack, gpu_owner
from scripts.verse_gpu_runtime import configure_cuda_cli


def main():
    pub = ROOT / "references/evidence/visual_sentence_search_v1"
    pub.mkdir(exist_ok=True)
    path = FOURTH / "confirmation.json"
    if sha(path) != read(ROOT/"references/evidence/fact_span_ranking_v1/reservation.json")["sha256"]:
        raise ValueError("Changed development questions")
    rows = read(path)
    baselines = unpack(FOURTH/"whole.json.gz")["original"]
    counters = defaultdict(Counter)
    results = []
    with gpu_owner(), torch.inference_mode():
        torch.set_float32_matmul_precision("highest")
        net = RankedAnswer("whole", "cuda")
        ink = Ink(net.palette)
        qs = [ink.encode(r["q"]) for r in rows]
        pages = [[ink.encode(b) for b in r["blocks"]] for r in rows]
        references = [[list(ink.encode(s)) for s in r["answers"]] for r in rows]
        tiles = ink.array()
        bank = torch.from_numpy(observations({"tiles":tiles},"ink")).cuda()
        for offset in range(0,len(qs),12):
            idx = range(offset,min(offset+12,len(qs)))
            batch = [qs[i] for i in idx]
            spans = net.select(bank,batch,[pages[i] for i in idx])
            strips = [f for _,_,f in spans]
            for i, fact in zip(idx, strips, strict=True):
                if sha256(tiles[np.array(fact,int)].tobytes()).hexdigest() != baselines[i]["fact_sha256"]:
                    raise ValueError("Reader changed the fact")
            ctx,h = net.writer.encode(bank,net.n,pack(batch,"cuda"),pack(strips,"cuda"))
            stops,parents,glyphs,bound = search(net.writer,ctx,h)
            best = []
            for j,i in enumerate(idx):
                score, flat = stops[:,j].reshape(-1).max(0)
                t,k = divmod(int(flat),stops.size(-1))
                pred = reconstruct(parents,glyphs,t,j,k)
                gold = references[i]
                offered = False
                for ref in gold:
                    n = len(ref)
                    if n >= len(stops):
                        continue
                    offered |= any(torch.isfinite(stops[n,j,k]) and reconstruct(parents,glyphs,n,j,k)==ref for k in range(stops.size(-1)))
                old = baselines[i]
                old_exact = old["stop"] and any(tiles[np.asarray(r,int)].tobytes().hex()==old["ink_hex"] for r in gold)
                exact = pred in gold
                certified = float(score)-float(bound[j]) > 1e-5
                if exact and not offered:
                    raise ValueError("Missing emitted reference")
                best.append(np.asarray([*pred,1],np.int32))
                counters[rows[i]["task"]].update(n=1,greedy=old_exact,search=exact,offered=offered,
                    certified=certified,certified_wrong=certified and not exact,
                    empty=not pred,gains=exact and not old_exact,losses=old_exact and not exact)
                results.append(dict(index=i, cells=len(pred), sha256=sha256(tiles[np.asarray(pred,int)].tobytes()).hexdigest(),
                                    exact=exact,offered=bool(offered),greedy_exact=old_exact,score=float(score),
                                    pruned_bound=float(bound[j]), certified=certified))
            replay = (-net.writer.loss(ctx,h,pack(best,"cuda")) * torch.tensor([len(x) for x in best],device="cuda")).tolist()
            for j,s in enumerate(replay):
                error = abs(results[offset+j]["score"]-s)
                results[offset+j]["score_replay_error"] = error
                if error>1e-3:
                    raise ValueError(f"Independent likelihood mismatch: {error}")
    dump(pub/"development.json",dict(summary=counters,records=results,cohort=sha(path),
        protocol=sha(ROOT/"references/visual_sentence_search_protocol.md"),
        implementation=sha(ROOT/"ilm/visual_lm/stopped_search.py")))
    print(dict(counters))


if __name__=="__main__":
    configure_cuda_cli()
    main()
