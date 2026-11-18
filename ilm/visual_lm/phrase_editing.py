"""Generate a complete translation with the learned image-based edit controller."""
import numpy as np
import torch

from ilm.visual_lm.idiom_ink import visual_input
from ilm.visual_lm.local_ink import cell_context, signatures
from ilm.visual_lm.visual_edit_writer import COPY, DROP, INSERT, REPLACE, STOP


@torch.inference_mode()
def write_sentence(net, picture, prompt, stream, pen, update, limit=192):
    """Return packed generated images, termination cause and number of insertion runs."""
    if picture.ndim != 2 or picture.shape[1] != 1024 or not len(picture) or limit < 1:
        raise ValueError('Supply a nonempty stream of 32-by-32 source cells and a positive limit')
    size=len(picture)
    source=visual_input(picture[None],net.landmarks)
    mem,initial=net.encode(source,torch.ones_like(source[:,:,0],dtype=torch.bool),
                           prompt,torch.ones_like(prompt[:,:,0],dtype=torch.bool))
    h=initial
    sig=signatures(picture[:,net.landmarks].cpu().numpy())
    output,generated=[],[]
    cursor=insertions=0
    after_phrase=False
    for _ in range(size+limit+2):
        ix=torch.tensor([cursor],device=h.device)
        eof=ix==size
        state=(mem[:,cursor]+h).tanh()
        actions=net.actions(state,eof)
        if after_phrase:
            actions[:,INSERT]=-torch.inf
        act=int(actions.argmax(1))
        if act==STOP:
            return output,'stop',insertions
        if act==DROP:
            cursor+=1
            after_phrase=False
            continue
        if len(output)==limit:
            return output,'sentence_limit',insertions
        ended=True
        if act==INSERT:
            context=np.array([cell_context(sig,cursor,np.asarray(generated,np.uint64))],dtype=np.uint64)
            onset=net.boundary(mem,ix.new_zeros(1),ix)
            marks,length,stop=net.complete(onset,h,initial,eof,stream=stream,local_context=context,
                interaction=pen,first_emission=update,limit=min(32,limit-len(output)))
            draw=marks[0,:int(length[0])]
            ended=bool(stop[0])
            insertions+=1
        elif act==COPY:
            draw=picture[cursor:cursor+1]
        elif act==REPLACE:
            bits=net.draw.choose(state)
            draw=net.motor(bits)>0
            draw[:,net.landmarks]=bits.bool()
        else:
            raise ValueError('Unexpected edit operation')
        for item in draw:
            bits=item[None,None,net.landmarks]
            value,_=net.advance_history(bits,initial,h[None].contiguous())
            h=value[:,0]
        code=draw.cpu().numpy()
        output.extend(np.packbits(code,axis=1))
        generated.extend(signatures(code[:,net.landmarks.cpu().numpy()]))
        if not ended:
            return output,'sentence_limit' if len(output)==limit else 'phrase_limit',insertions
        after_phrase=act==INSERT
        cursor+=act!=INSERT
    raise RuntimeError('Non-progressing edit trajectory')
