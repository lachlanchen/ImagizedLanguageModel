"""Generate physical codes while summing compatible latent source/edit positions."""
import torch
from torch.nn import functional as F

from ilm.visual_lm.idiom_ink import bit_image, nibbles, visual_input
from ilm.visual_lm.visual_edit_writer import COPY, DROP, INSERT, REPLACE, STOP


def next_code(net, state, source_codes, copy_mass, insert_mass, replace_mass):
    b,s,_ = state.shape
    base = net.draw.plan(state)
    prefix = base.new_zeros(b,1,base.shape[-1])
    pattern = base.new_ones(b,1,s)
    compatible = copy_mass[:,None]
    history = source_codes.new_empty(b,1,0)
    pruned = base.new_zeros(b)
    rows = torch.arange(b,device=state.device)[:,None]
    for j in range(source_codes.shape[-1]):
        logits = net.draw.output(F.gelu(base[:,None] + prefix[:,:,None] + net.draw.position[j])) + net.draw.bias[j]
        extended = pattern[...,None]*logits.softmax(-1)
        masses = ((insert_mass+replace_mass)[:,None,:,None]*extended).sum(2)
        current = source_codes[:,None,:,j].expand(-1,compatible.shape[1],-1)
        masses.scatter_add_(2,current,compatible)
        values,flat = masses.flatten(1).topk(5,dim=1)
        pruned = torch.maximum(pruned,values[:,4])
        parent, code = flat[:,:4]//16, flat[:,:4]%16
        prefix = prefix[rows,parent] + net.draw.prefix[j,code]
        pattern = extended[rows,parent].gather(-1,code[:,:,None,None].expand(-1,-1,s,1)).squeeze(-1)
        compatible = compatible[rows,parent]*(current[rows,parent]==code[:,:,None])
        history = torch.cat((history[rows,parent],code[:,:,None]),-1)
    return bit_image(history[:,0]),pattern[:,0],compatible[:,0],values[:,0],pruned


@torch.inference_mode()
def generate_prefix(net, source, lengths, question, qlen, *, max_output=192):
    if max_output<1 or (lengths<1).any() or (qlen<1).any():
        raise ValueError('A nonempty source, question and positive output limit are required')
    device = source.device
    b,n,_ = source.shape
    index = torch.arange(n+1,device=device)[None]
    rows = torch.arange(b,device=device)
    mem, initial = net.encode(visual_input(source,net.landmarks),index[:,:-1]<lengths[:,None],
                              question, torch.arange(question.shape[1],device=device)[None]<qlen[:,None])
    codes = F.pad(nibbles(source[:,:,net.landmarks]),(0,0,0,1))
    valid = index <= lengths[:,None]
    eof = index==lengths[:,None]
    alpha = mem.new_zeros(b,n+1)
    alpha[:,0] = 1.
    state = initial
    out = torch.zeros(b,max_output,1024,device=device,dtype=torch.bool)
    count = torch.zeros_like(lengths)
    stopped = torch.zeros_like(lengths,dtype=torch.bool)
    logp = initial.new_zeros(b,dtype=torch.float64)
    uncertified = torch.zeros_like(lengths)
    residual = torch.zeros_like(logp)
    for step in range(max_output+1):
        live = ~stopped
        h = (mem + state[:,None]).tanh()
        action = net.actions(h,eof)
        delete = torch.where(index<lengths[:,None],action[:,:,DROP],0.).double()
        offset = F.pad(delete[:,:-1].cumsum(1),(1,0))
        arrival = (offset + torch.logcumsumexp(alpha.double().log()-offset,1)).exp().float()*valid
        weights = action.exp()*arrival[:,:,None]
        copy,replace,insert,stop = (weights[:,:,k] for k in (COPY,REPLACE,INSERT,STOP))
        end_prob = stop.sum(1)
        residual = torch.maximum(residual,torch.where(live,((copy+replace+insert).sum(1)+end_prob-1).abs(),0.))
        bits,pattern,copied,mass,bound = next_code(net,h,codes,copy,insert,replace)
        end = end_prob>=mass
        cert = torch.maximum(mass,end_prob)>=bound
        uncertified += live & ~cert
        stopped |= live & end
        logp += torch.where(live & end,end_prob.double().log(),0.)
        emit = live & ~end
        if step==max_output or not emit.any():
            break
        # Copy the highest-weight source instance or draw the same physical code.
        weight, position = copied[:,:n].max(1)
        drawn = net.motor(bits)>0
        drawn[:,net.landmarks] = bits.bool()
        image = torch.where((weight>((insert+replace)*pattern).sum(1))[:,None],source[rows,position],drawn)
        out[rows[emit],step] = image[emit]
        count += emit
        logp += torch.where(emit,mass.double().log(),0.)
        updated = insert*pattern + F.pad((copied+replace*pattern)[:,:-1],(1,0))
        alpha = torch.where(emit[:,None],updated/mass[:,None].clamp_min(1e-30),alpha)
        new,_ = net.advance_history(image[:,None,net.landmarks],initial,state[None].contiguous())
        state = torch.where(emit[:,None],new[:,0],state)
    return out,count,stopped,logp,uncertified,residual
