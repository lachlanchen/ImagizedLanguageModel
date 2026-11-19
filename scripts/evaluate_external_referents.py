"""Free image completion on the pinned Chinese XWinograd test split."""

import argparse
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd
from safetensors.torch import load_file
import torch

from scripts.evaluate_same_passage_writer import image_archive
from scripts.train_pictured_participants import (
    PUBLIC, ROOT, START, WORK, Tiles, load, load_visual_writer,
    memory_bank, physical_tables, sha, write_json,
)

REVISION = "90b619ef5278605cd4f572e3e8506ab16afd44f2"
DIGEST = "ec37734821dd4f9fd6e49aa40d4043c15ee9225a7c799ce06a42922278cdf307"
SOURCE = f"https://huggingface.co/datasets/Muennighoff/xwinograd/resolve/{REVISION}/zh/test-00000-of-00001.parquet"
DATA = ROOT / "data/evaluation_only/xwinograd_zh_v1/test.parquet"
QUESTION = "句中空格指的是谁或什么？请写出所指的人或事物。"


@torch.inference_mode()
def run(fetch):
    if not DATA.exists() and fetch:
        DATA.parent.mkdir(parents=True, exist_ok=True)
        from hashlib import sha256
        with urllib.request.urlopen(SOURCE, timeout=30) as response:
            payload = response.read()
        if sha256(payload).hexdigest() != DIGEST:
            raise ValueError("Dataset digest mismatch")
        DATA.write_bytes(payload)
    assert sha(DATA) == DIGEST
    df = pd.read_parquet(DATA)
    assert len(df) == 504
    bank = memory_bank()
    tile = Tiles(bank)
    ps, qs, targets, rivals = [], [], [], []
    def pixels(seq):
        return b"".join(bank.tiles[k] for k in seq)
    for row in df.itertuples():
        p = tile(row.sentence.strip())
        ps.append(p)
        qs.append(tile(QUESTION))
        options = [pixels(tile(x.strip())) for x in (row.option1, row.option2)]
        y = int(row.answer)-1
        targets.append(options[y])
        rivals.append(options[1-y])
    identical = sum(t == r for t,r in zip(targets,rivals,strict=True))
    from scripts.train_grounded_glyph_writer import padded
    device = "cpu"
    writer = load_visual_writer(START,device).requires_grad_(False)
    feat, mem, _ = physical_tables(bank,device)
    ink = writer.ink(feat)
    keys = writer.write_key(ink[mem])
    report = {}
    for arm in ("parent","matched","exchanged"):
        from ilm.visual_lm.participant_reader import ParticipantReader
        from scripts.train_pictured_participants import PRIOR
        net = ParticipantReader().eval()
        if arm=="parent":
            missing,extra=net.load_state_dict(load_file(str(PRIOR)),strict=False)
            assert not extra and set(missing)=={k for k,v in net.named_parameters() if v.requires_grad}
        else:
            path=WORK/arm/"reader.safetensors"
            assert sha(path)==load(WORK/arm/"fit.json")["weights"]
            net.load_state_dict(load_file(str(path)))
        imgs, done = [], []
        for i in range(0,len(qs),16):
            p,q = (torch.as_tensor(padded(x[i:i+16])) for x in (ps,qs))
            out,stop = net.generate(writer,ink,p,q,keys,mem)
            imgs.extend(pixels(row[row!=0]) for row in out.numpy())
            done.extend(stop.tolist())
        exact = sum(s and z==t for s,z,t in zip(done,imgs,targets,strict=True))
        other = sum(s and z==r and t!=r for s,z,t,r in zip(done,imgs,targets,rivals,strict=True))
        archive = WORK/arm/"xwinograd.npz"
        image_archive(archive,imgs,done)
        report[arm] = dict(exact=exact,rival=other,neither=len(ps)-exact-other,
                           stopped=sum(done),archive=sha(archive))
    boundary_match = lambda text, target: any(text[j:j+len(target)]==target for j in range(0,len(text),128))
    write_json(PUBLIC/"external_referents.json",dict(source=SOURCE,source_sha256=DIGEST,
        release_license="CC BY 4.0",rows=len(ps),question=QUESTION,arms=report,
        target_in_passage=sum(boundary_match(pixels(p),t) for p,t in zip(ps,targets,strict=True)),
        equal_option_images=identical, blank_counts=df.sentence.str.count("_").value_counts().to_dict(),
        extra_choices_supplied=False,fit_updates=0,runner=sha(Path(__file__))))
    print(report,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch",action="store_true")
    args=parser.parse_args()
    torch.set_num_threads(2)
    if (PUBLIC/"external_referents.json").exists():
        raise FileExistsError("Frozen external evaluation already exists")
    run(args.fetch)
