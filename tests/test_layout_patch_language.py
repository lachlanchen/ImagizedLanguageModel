import numpy as np

from ilm.visual_lm.layout_patch_language import LayoutPatchLanguage
from ilm.visual_lm.patch_count_language import assemble, tile_features


def test_shared_layout_normalization_feedback_and_causal_end(tmp_path):
    cells = np.zeros((4,32,32),np.uint8)
    for i in range(1,4):
        cells[i,(i-1)*8:i*8,8:16] = 1
    passages = [[1,0,2]]*40 + [[3,0,1]]*40
    model = LayoutPatchLanguage.fit(cells, passages)
    model.save(tmp_path/'model.npz')
    restored = LayoutPatchLanguage.load(tmp_path/'model.npz')
    signatures, bits = tile_features(cells)
    assert np.array_equal(np.stack([assemble(b) for b in bits]), cells)
    assert model.ending[-1].counts.sum() == 240
    for history, target in (([1,0],2), ([3,0],1)):
        prompt = cells[history]
        probs = model.layout_probabilities(model.history(prompt))
        assert np.isclose(probs.sum(),1) and np.all(probs > 0)
        for j in range(16):
            patches = model.patch_probabilities(j, int(signatures[target]), 0)
            assert np.isclose(patches.sum(),1) and np.all(patches > 0)
        for actor in [model, restored]:
            out, stopped = actor.generate(prompt, maximum_new=4)
            assert stopped and np.array_equal(out, cells[[target]])
        for actor in [model, restored]:
            out, _ = actor.generate(prompt, generator=np.random.default_rng(112))
            assert np.array_equal(out,cells[[target]])
