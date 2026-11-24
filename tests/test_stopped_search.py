from itertools import product

import torch

from ilm.visual_lm.region_writer import RegionWriter
from ilm.visual_lm.stopped_search import reconstruct, search


@torch.inference_mode()
def test_exact_probability_bound_on_all_stopped_streams():
    torch.manual_seed(19)
    writer = RegionWriter(width=8).eval()
    features = torch.randn(7, 256)
    question = torch.tensor([[2, 3], [4, 2]])
    fact = torch.tensor([[4, 5], [3, 6]])
    ctx, h = writer.encode(features, 6, question, fact)
    stops, parents, glyphs, bound = search(writer, ctx, h, width=2, limit=3)
    legal = [word+(1,) for n in range(3) for word in product(range(2,7), repeat=n)]
    for j in range(2):
        ids = torch.zeros(len(legal),3,dtype=torch.long)
        for k, word in enumerate(legal):
            ids[k,:len(word)] = torch.tensor(word)
        qs = question[j:j+1].expand(len(legal),-1)
        facts = fact[j:j+1].expand(len(legal),-1)
        full, hidden = writer.encode(features,6,qs,facts)
        independent = -writer.loss(full,hidden,ids) * ids.ne(0).sum(1)
        visited = set()
        for t in range(3):
            for b in range(2):
                if torch.isfinite(stops[t,j,b]):
                    word = tuple(reconstruct(parents,glyphs,t,j,b))+(1,)
                    visited.add(word)
                    assert abs(float(stops[t,j,b]-independent[legal.index(word)])) < 2e-5
        outside = [i for i,w in enumerate(legal) if w not in visited]
        assert independent[outside].max() <= bound[j]+2e-5
        if stops[:,j].max() >= bound[j]:
            assert abs(stops[:,j].max()-independent.max()) < 2e-5
