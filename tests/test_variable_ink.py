import numpy as np
import pytest
from ilm.visual_lm.variable_ink import bind_card,expand
from scripts.train_variable_ink import MODEL,bindings
from scripts.write_semantic_page import read_card


def test_swapping_unseen_physical_spans_keeps_the_same_question_plan():
    p,anchors=bindings()
    before,src=bind_card(read_card(MODEL/"examples/historical-original-card.png"),p,anchors)
    after,other=bind_card(read_card(MODEL/"examples/historical-swapped-card.png"),p,anchors)
    assert np.array_equal(before,after)
    assert np.array_equal(src["name"],other["near"])
    assert not np.array_equal(src["name"],src["near"])
    alphabet=np.zeros((2,128),np.uint8)
    assert np.array_equal(expand([2,1,3],alphabet,src),np.concatenate((src["name"],alphabet[1:],src["near"])))


def test_absent_and_multiple_referents_are_rejected():
    p,anchors=bindings();data=read_card(MODEL/"examples/latin-original-card.png")
    with pytest.raises(ValueError,match="More than one"):
        bind_card(np.concatenate((data,data)),p,anchors)
    with pytest.raises(ValueError,match="unavailable"):
        expand([2],np.zeros((2,128),np.uint8),{})
