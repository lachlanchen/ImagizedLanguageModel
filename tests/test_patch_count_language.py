import numpy as np
from ilm.visual_lm.patch_count_language import PatchCountLanguage, assemble, tile_features


def test_bit_order_support_and_roundtrip(tmp_path):
    rng = np.random.default_rng(81)
    tiles = rng.integers(0,2,(9,32,32),dtype=np.uint8)
    _, parts = tile_features(tiles)
    assert np.array_equal(tiles, np.stack([assemble(r) for r in parts]))
    data = [np.array([0,3,1,4,0,5,1]),np.array([0,3,6]),np.array([8])]
    model = PatchCountLanguage.fit(tiles,data)
    model.save(tmp_path/'weights.npz')
    restored=PatchCountLanguage.load(tmp_path/'weights.npz')
    hist=model.history(tiles[:2])
    for j,part in enumerate(model.parts):
        p=model.patch_probabilities(j,hist,0)
        assert (p > 0).all() and np.isclose(p.sum(),1)
    assert np.array_equal(model.draw(hist),restored.draw(hist))
    assert 0 < restored.stop_probability(hist) < 1
    a,_=model.generate(tiles[:1],generator=np.random.default_rng(5))
    b,_=restored.generate(tiles[:1],generator=np.random.default_rng(5))
    assert np.array_equal(a,b)


def test_causal_predictions_use_real_transitions_and_endings():
    tiles=np.zeros((4,32,32),np.uint8)
    for j in range(1,4):
        tiles[j,(j-1)*8:j*8,:8]=1
    data=[np.array([1,0,2])]*50 + [np.array([3,0,1])]*50
    model=PatchCountLanguage.fit(tiles,data)
    for prompt,target in (([1,0],2),([3,0],1)):
        images,stopped=model.generate(tiles[prompt],maximum_new=4)
        assert stopped and len(images)==1 and np.array_equal(images[0],tiles[target])
