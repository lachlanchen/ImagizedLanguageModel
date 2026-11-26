"""Autoregressive translation from pictured cells through variable-length drawn spans."""
from collections import Counter
from dataclasses import dataclass
import gzip
import json
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from huggingface_hub import snapshot_download
from transformers import MarianMTModel

from ilm.visual_lm.field_to_glyph import ROOT, load_weights
from ilm.visual_lm.glyph_measure import weighted_cross_attention
from ilm.visual_lm.optical_state_reader import OpticalStateReader
from scripts.write_neural_glyphs import load_model
from scripts.write_ordered_translation import MODEL as PALETTE, sha

COUNTS = ROOT / "references/evidence/optical_span_rollout_v1"


@dataclass
class SpanProjection:
    keys: torch.Tensor
    bias: torch.Tensor
    key_index: torch.Tensor
    surface_index: torch.Tensor
    frequency: torch.Tensor

    def select(self, hidden, number):
        # The stop occupies surface 0. A silent internal span still advances the model.
        values = (hidden @ self.keys.T + self.bias).softmax(-1)
        contributions = values[:, self.key_index] * self.frequency
        mass = values.new_zeros((len(hidden), number))
        mass.index_add_(1, self.surface_index, contributions)
        return mass.argmax(1)


def index_tensor(values, device):
    return torch.tensor(values, device=device, dtype=torch.long)


def make_projection(entries, phase, surfaces, teacher, device):
    matches = [(k,surfaces[tuple(s)],n) for when,k,s,n in entries if when==phase]
    stop = teacher.config.eos_token_id
    matches.append((stop,0,1))
    total = Counter()
    for k,_,n in matches:
        total[k] += n
    keys = sorted(total)
    idx = {k:i for i,k in enumerate(keys)}
    return SpanProjection(teacher.lm_head.weight[keys].detach().to(device),
        teacher.final_logits_bias[0,keys].detach().to(device),
        index_tensor([idx[k] for k,_,_ in matches],device),
        index_tensor([s for _,s,_ in matches],device),
        torch.tensor([n/total[k] for k,_,n in matches],device=device))


class OpticalSpanWriter:
    @torch.inference_mode()
    def __init__(self, lang, device):
        direction = ("en","zh").index(lang)
        receipt = json.loads((COUNTS/(lang+"-fit.json")).read_text())
        fitted = COUNTS/(lang+"-span-counts.json.gz")
        if sha(fitted)!=receipt["weights"]:
            raise ValueError("Optical span counts changed")
        entries = json.loads(gzip.decompress(fitted.read_bytes()))
        path = snapshot_download(receipt["model"],revision=receipt["revision"],local_files_only=True)
        teacher = MarianMTModel.from_pretrained(path,local_files_only=True,attn_implementation="eager").eval()
        self.source_parameters = sum(p.numel() for p in teacher.parameters())
        self.device = torch.device(device)
        _, weights = load_weights(ROOT/"models/readout_objective"/lang/"prediction")
        self.reader = OpticalStateReader().eval().to(device)
        self.reader.load_state_dict(weights)
        decoder = teacher.model.decoder
        start = teacher.config.decoder_start_token_id
        native_embeddings = decoder.embed_tokens.weight.detach()
        self.begin = (native_embeddings[start]*decoder.embed_scale).to(device)
        sequences = sorted({tuple(s) for _,_,s,_ in entries})
        surfaces = {s:i+1 for i,s in enumerate(sequences)}
        self.bank = [make_projection(entries,i,surfaces,teacher,device) for i in (0,1)]
        sums = torch.zeros((len(sequences)+1,len(self.begin)))
        totals = torch.zeros(len(sums))
        for j in range(0,len(entries),1024):
            rows = entries[j:j+1024]
            ids = index_tensor([k for _,k,_,_ in rows],"cpu")
            ix = index_tensor([surfaces[tuple(s)] for _,_,s,_ in rows],"cpu")
            n = torch.tensor([n for _,_,_,n in rows],dtype=sums.dtype)
            totals.index_add_(0,ix,n)
            sums.index_add_(0,ix,n[:,None]*native_embeddings[ids])
        self.feedback = (decoder.embed_scale*sums/totals.clamp_min(1)[:,None]).to(device)
        self.feedback[0] = self.begin
        self.decoder = decoder
        # No vocabulary lookup or text output projection is called during deployment.
        self.decoder.embed_tokens = nn.Identity()
        self.decoder.to(device)
        self.attention_owner = SimpleNamespace(model=SimpleNamespace(decoder=self.decoder))
        inventory = json.loads((PALETTE/"inventories.json").read_text())[direction][1:]
        palette = np.load(PALETTE/"palette.npy")
        motor = load_model(device)
        codes = getattr(motor, f"codes_{direction}")
        self.motor_parameters = sum(p.numel() for p in motor.motor.parameters())
        self.ink = torch.cat([motor.motor(b).gt(0).reshape(-1,32,32) for b in codes.split(256)])
        idx = {v:i for i,v in enumerate(inventory)}
        self.shapes = [()] + [tuple(idx[v] for v in seq) for seq in sequences]
        # Re-identify the physically emitted images to obtain feedback. This enforces
        # the assumption of exact optical recovery used in the recurrence.
        emitted = np.packbits(self.ink.cpu().numpy().reshape(len(codes),1024),axis=1)
        lookup = {bytes(row):v for v,row in zip(inventory,palette[inventory],strict=True)}
        read_ink = [lookup.get(bytes(row)) for row in emitted]
        if read_ink != inventory:
            raise ValueError("Neural motor failed the calibrated pixel readback")
        self.readback = [0] + [surfaces[tuple(read_ink[g] for g in shape)] for shape in self.shapes[1:]]
        self.cost = dict(pretrained_full_at_load=self.source_parameters,
                         retained_decoder=sum(p.numel() for p in self.decoder.parameters()),
                         optical_reader=sum(p.numel() for p in self.reader.parameters()),
                         motor_parameters_before_compilation=self.motor_parameters,
                         directional_keys_and_bias=sum(p.keys.numel()+p.bias.numel() for p in self.bank),
                         span_feedback_entries=self.feedback.numel(), compiled_ink_bytes=self.ink.numel(),
                         surface_types=len(sequences),lexical_keys=[len(p.keys) for p in self.bank])

    @torch.inference_mode()
    def write(self, grids, limit=128):
        if not grids or any(len(x)<1 or len(x)>96 or x.shape[1:]!=(32,32) for x in grids):
            raise ValueError("A source requires 1–96 aligned 32-square ink cells")
        if limit<1 or limit>128:
            raise ValueError("The supported generation budget is 1–128")
        images = [torch.as_tensor(x,device=self.device,dtype=torch.float32) for x in grids]
        lengths = [len(x) for x in images]
        flat = F.avg_pool2d(torch.cat(images)[:,None],2).flatten(1)
        handles = index_tensor(range(1,len(flat)+1),self.device).split(lengths)
        source = nn.utils.rnn.pad_sequence(handles,batch_first=True)
        field, mask, mass = self.reader.states(F.pad(flat,(0,0,1,0)),source)
        n = len(grids)
        outputs = [[] for _ in grids]
        boundaries = [[] for _ in grids]
        stopped = [False]*n
        finished = [False]*n
        x = self.begin.expand(n,1,-1)
        past = None
        with weighted_cross_attention(self.attention_owner,mass):
            for step in range(limit+1):
                layer = self.decoder(inputs_embeds=x,encoder_hidden_states=field,
                                     encoder_attention_mask=mask,use_cache=True,past_key_values=past)
                past = layer.past_key_values
                choices = self.bank[int(step>0)].select(layer.last_hidden_state[:,0],len(self.shapes)).tolist()
                for b,c in enumerate(choices):
                    if finished[b]: continue
                    if c==0:
                        stopped[b]=finished[b]=True
                    elif step==limit:
                        finished[b]=True
                    else:
                        tail = self.shapes[c]
                        room = limit-len(outputs[b])
                        outputs[b].extend(tail[:room])
                        boundaries[b].append(len(outputs[b]))
                        if len(tail)>room: finished[b]=True
                if all(finished): break
                x = self.feedback[index_tensor([self.readback[c] for c in choices],self.device)][:,None]
        return [(self.ink[index_tensor(line,self.device)].cpu().numpy(),end,edges)
                for line,end,edges in zip(outputs,stopped,boundaries,strict=True)]
