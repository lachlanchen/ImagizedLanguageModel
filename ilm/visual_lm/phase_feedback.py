"""Preserve the writing phase when reading an emitted span back as a decoder embedding."""
from collections import defaultdict
import gzip
import json
import torch
from huggingface_hub import snapshot_download
from transformers import MarianMTModel
from ilm.visual_lm.pictured_path_writer import PicturedPathWriter, COUNTS


class PhaseFeedbackWriter(PicturedPathWriter):
    @torch.inference_mode()
    def __init__(self, lang, device, *, pooled_control=False):
        super().__init__(lang,device)
        cfg=json.loads((COUNTS/(lang+'-fit.json')).read_text())
        entries=json.loads(gzip.decompress((COUNTS/(lang+'-span-counts.json.gz')).read_bytes()))
        sequences=sorted({tuple(span) for _,_,span,_ in entries})
        weights=defaultdict(lambda:defaultdict(dict))
        for phase,key,span,count in entries:
            table=weights[tuple(span)][phase]
            table[key]=table.get(key,0)+count
        # Surfaces occurring in just one phase already have exactly the registered mean.
        ambiguous=[(j,weights[span]) for j,span in enumerate(sequences,1)
                   if len(weights[span])==2 and len(set(weights[span][0])|set(weights[span][1]))>1]
        path=snapshot_download(cfg['model'],revision=cfg['revision'],local_files_only=True)
        native=MarianMTModel.from_pretrained(path,local_files_only=True,attn_implementation='eager').eval()
        decoder=native.model.decoder
        bank=decoder.embed_tokens.weight.detach()
        def mean(counts):
            ids=sorted(counts)
            fractions=torch.tensor([counts[k] for k in ids],dtype=bank.dtype)
            return decoder.embed_scale*(fractions @ bank[ids])/fractions.sum()
        original=len(self.shapes)
        initial_routing=torch.arange(original,device=self.device)
        feedback=self.feedback.clone()
        extras=[]
        for j,counts in ambiguous:
            alias=len(self.shapes)
            initial_routing[j]=alias
            # Aliases have the exact same physical output; phase plus read-back pixels determine the new address.
            assert self.readback[j]==j
            self.shapes.append(self.shapes[j])
            self.readback.append(alias)
            extras.append(feedback[j].clone() if pooled_control else mean(counts[0]).to(device))
            if not pooled_control:
                feedback[j]=mean(counts[1]).to(device)
        if extras:
            self.feedback=torch.cat((feedback,torch.stack(extras)))
            self.bank[0].surface_index=initial_routing[self.bank[0].surface_index]
        self.phase_cost=dict(aliased_shapes=len(extras),extra_feedback_bytes=len(extras)*feedback.shape[1]*feedback.element_size())
