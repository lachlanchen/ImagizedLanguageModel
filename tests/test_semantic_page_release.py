"""Regression checks for the optical entry point, counterfactual and scorer ambiguity."""
import numpy as np
import pytest
import torch
from PIL import Image
from ilm.visual_lm.semantic_listener import beam_write
from scripts.evaluate_semantic_listener import canonical
from scripts.train_semantic_page import MODEL, load, palette, read
from scripts.train_semantic_listener import load_listener
from scripts.write_semantic_page import read_card, render, printed


def test_paired_pngs_reverse_policy_with_same_writer():
    torch.set_num_threads(2)
    writer=load("ink","cpu"); listener=load_listener("cpu")
    examples=MODEL/"examples"
    before=read_card(examples/"adult-card.png")
    after=read_card(examples/"family-card.png")
    assert np.any(before!=after,axis=(1,2)).sum()==1
    for label,packed in (("adult",before),("family",after)):
        ids,stopped=render(writer,packed,"cpu",listener)
        assert stopped
        ink=printed(np.load(MODEL/"palette.npy")[ids])
        with Image.open(examples/(label+"-answer.png")) as expected:
            assert np.array_equal(np.asarray(ink),np.asarray(expected))


def test_meaning_aliases_do_not_merge_polarity():
    assert canonical({("customer rating","5 out of 5")})=={("customer rating","high")}
    assert canonical({("familyFriendly","yes")})!=canonical({("familyFriendly","no")})


def test_empty_card_is_rejected():
    torch.set_num_threads(2)
    with pytest.raises(ValueError,match="Invalid card"):
        render(load("ink","cpu"),np.zeros((1,40,128),np.uint8),"cpu",load_listener("cpu"))
