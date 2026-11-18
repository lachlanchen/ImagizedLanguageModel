"""Predict a finite insertion allowance at each visible source position."""
import torch
from torch import nn

from ilm.visual_lm.visual_edit_writer import VisualEditWriter


class FertilityEditor(VisualEditWriter):
    def __init__(self,landmarks,max_count,width=128):
        super().__init__(landmarks,width)
        self.fertility = nn.Sequential(nn.Linear(width*2,width),nn.GELU(),nn.Linear(width,max_count+1))

    def count_logits(self,memory,global_state):
        context = global_state[:,None].expand(-1,memory.shape[1],-1)
        return self.fertility(torch.cat((memory,context),dim=-1))

    def insertion_counts(self,memory,global_state):
        return self.count_logits(memory,global_state).argmax(-1)

    def generate(self,*args,**kwargs):
        return super().generate(*args,insertion_limit=self.insertion_counts,**kwargs)
