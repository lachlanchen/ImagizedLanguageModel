import numpy as np

from ilm.visual_lm.grounded_detail_writer import GroundedDetailWriter, condition, fit_modes


def test_weighted_modes_and_ties():
    keys = condition([1,1,1,2,2],[0,0,0,1,1],5)
    table, modes = fit_modes(keys,np.array([7,7,5,3,1]),np.array([2,4,5,3,3]))
    assert table['layout'].tolist()==[1,2] and modes.tolist()==[7,1]


def test_position_order_and_reloaded_drawing(tmp_path):
    gen = np.random.default_rng(821)
    source = gen.integers(0,256,size=(17,128),dtype=np.uint8)
    target = np.bitwise_xor(source,0b10010001)
    model = GroundedDetailWriter.fit(source,target,np.arange(1,18))
    assert np.array_equal(model.paint(source),target)
    model.save(tmp_path/'motor.npz')
    reloaded = GroundedDetailWriter.load(tmp_path/'motor.npz')
    assert np.array_equal(reloaded.paint(source),target)
    assert np.array_equal(reloaded.paint(source,detail=False),model.paint(source,detail=False))
