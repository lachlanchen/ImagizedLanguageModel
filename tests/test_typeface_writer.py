import torch
from scripts.typeface_writer_cases import Styled
from ilm.visual_lm.causal_glyph_writer import CausalGlyphWriter


def test_copy_and_fixed_ink_can_share_one_accepted_answer():
    a,b,c,d=(bytes([i])*128 for i in range(4))
    case=Styled("b",b"",b"",(a+b,c+a),(c+d,b+c),"id")
    assert case.grade(c+b,True)==(False,False,True)
    assert case.grade(c+b,False)==(False,False,False)
    assert case.grade(a+a,True)==(False,False,False)
    assert case.grade(b"",True)==(False,False,False)


def test_mapping_changes_read_state_only():
    class Mapper:
        def identify(self, raw):
            return torch.tensor([2,1]), torch.tensor([True,False])
    class Writer:
        typeface_map=Mapper()
        def encode(self, raw):
            return raw.float()
    cells=torch.tensor([[8,7],[6,5]],dtype=torch.uint8)
    alphabet=torch.tensor([[0,0],[1,2],[3,4]],dtype=torch.float32)
    result=CausalGlyphWriter.read_extra(Writer(),cells,alphabet)
    assert result.tolist()==[[3,4],[6,5]]
    assert cells.tolist()==[[8,7],[6,5]]
    assert alphabet.tolist()==[[0,0],[1,2],[3,4]]
