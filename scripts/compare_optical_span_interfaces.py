"""Control the source and output interfaces around the same inherited decoder."""
import argparse
import json
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
from huggingface_hub import snapshot_download
import torch
from transformers import MarianMTModel, MarianTokenizer
from transformers.modeling_outputs import BaseModelOutput
from ilm.visual_lm.field_to_glyph import load_weights
from ilm.visual_lm.glyph_measure import weighted_cross_attention
from ilm.visual_lm.optical_state_reader import OpticalStateReader
from scripts.evaluate_optical_state_reader import as_answer
from scripts.evaluate_optical_span_rollout import (ARCHIVE, PUB, ROOT, RUN, blind_votes, configure_cuda_cli,
    dump, gpu_owner, grid_bits, read, requests, sha, summarize, work_interval)


@torch.inference_mode()
def controls(rows):
    out=[{} for _ in rows]
    with ZipFile(ARCHIVE) as z:
        grids=[grid_bits(BytesIO(z.read(f"{i:04d}/source.png")),len(r["input"])) for i,r in enumerate(rows)]
        records=json.loads(z.read("records.json"))
    for t,lang in enumerate(("en","zh")):
        fit=read(PUB/(lang+"-fit.json"))
        path=snapshot_download(fit["model"],revision=fit["revision"],local_files_only=True)
        native=MarianMTModel.from_pretrained(path,local_files_only=True,attn_implementation="eager").eval()
        tok=MarianTokenizer.from_pretrained(path,local_files_only=True)
        reader=OpticalStateReader().eval()
        reader.load_state_dict(load_weights(ROOT/"models/readout_objective"/lang/"prediction")[1])
        ix=[i for i,r in enumerate(rows) if r["t"]==t]
        settings=dict(do_sample=False,num_beams=1,max_new_tokens=128,forced_eos_token_id=None)
        for j in range(0,len(ix),12):
            group=ix[j:j+12]
            text=tok([rows[i]["source"] for i in group],padding=True,return_tensors="pt")
            words=native.generate(**text,**settings)
            inputs=[torch.from_numpy(grids[i]).float() for i in group]
            flat=torch.nn.functional.avg_pool2d(torch.cat(inputs)[:,None],2).flatten(1)
            handles=torch.arange(1,len(flat)+1).split(list(map(len,inputs)))
            batch=torch.nn.utils.rnn.pad_sequence(handles,batch_first=True)
            f,m,bias=reader.states(torch.nn.functional.pad(flat,(0,0,1,0)),batch)
            with weighted_cross_attention(native,bias):
                pictured=native.generate(encoder_outputs=BaseModelOutput(last_hidden_state=f),attention_mask=m,**settings)
            for b,i in enumerate(group):
                out[i]=dict(native=as_answer(words[b],tok),pictured=as_answer(pictured[b],tok),
                            span=records[i]["answers"]["span"],reference=dict(text=rows[i]["ref"],stop=True),
                            foil=dict(text=rows[i]["foil"],stop=True))
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grade",action="store_true")
    args=parser.parse_args()
    torch.set_num_threads(2)
    rows=requests()
    path=PUB/"interfaces.json"
    binding=dict(archive=sha(ARCHIVE),program=sha(Path(__file__)),protocol=sha(ROOT/"references/optical_span_control_protocol.md"))
    if not args.grade:
        if path.exists():
            assert read(path)["binding"]==binding
            return
        dump(path,dict(binding=binding,answers=controls(rows)))
        return
    configure_cuda_cli()
    assert read(path)["binding"]==binding
    with gpu_owner():
        out=PUB/"interface-votes.json"
        if out.exists():
            assert read(out)["inputs"]==sha(path)
            return
        answers=read(path)["answers"]
        votes=blind_votes(rows,answers)
        primary=summarize(rows,answers,votes)
        intervals={name:work_interval(primary["works"],a,b) for name,a,b in [
            ("source","native","pictured"),("writer","pictured","span")]}
        dump(out,dict(inputs=sha(path),records=primary,votes=votes,intervals=intervals))
        print(primary["total"],intervals,flush=True)


if __name__=="__main__":
    main()
