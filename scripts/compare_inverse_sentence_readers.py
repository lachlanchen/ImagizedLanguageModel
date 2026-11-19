"""Apply the same annotation-free visual sentence selector to three frozen readers."""
import fcntl
import numpy as np
import torch
from safetensors.torch import load_model

from ilm.visual_lm.ordered_passage_planner import OrderedPassageComposer
from ilm.visual_lm.pictured_sentence import locate
from scripts.audit_traditional_passage_training import archived, clustered_delta
from scripts.evaluate_same_passage_writer import generate, image_archive, reference_sets
from scripts.measure_natural_role_questions import BASE
from scripts.train_inverse_role_writer import (DEVICE, PUBLIC, ROOT, WHO, WORK, PairCorpus,
    Tiles, load, load_visual_writer, memory_bank, physical_tables, sha, write_json)


@torch.inference_mode()
def compare():
    torch.set_num_threads(2)
    study = load(PUBLIC/"evidence_sentences.json")
    idx = study["question_indices"]
    bank = memory_bank()
    ds = PairCorpus(bank,"development")
    boundaries = [bank.tiles[k] for k in Tiles(bank)("。！？!?")]
    def as_grid(ids):
        return np.frombuffer(b"".join(bank.tiles[k] for k in ids),np.uint8).reshape(-1,128)
    qs = [ds.questions[i] for i in idx]
    selected = []
    for i,q in zip(idx,qs,strict=True):
        ps = ds.passages[ds.groups[i]]
        a,b = locate(as_grid(ps),as_grid(q),boundaries)
        selected.append(ps[a:b])
    refs = reference_sets(ds,idx,bank)
    groups = ds.groups[idx] if isinstance(ds.groups,np.ndarray) else np.array([ds.groups[i] for i in idx])
    with np.load(WHO/"cohort.npz") as split:
        pairs = split["development_pairs"]
    def correct(images,ends):
        return np.array([bool(s and p in ref) for p,s,ref in zip(images,ends,refs,strict=True)])
    evidence = WORK/"evidence_sentences"
    out = WORK/"sentence_controls";out.mkdir()
    matched = correct(*archived(evidence/"visual_overlap.npz",study["conditions"]["visual_overlap"]["archive"]))
    result = {}
    for arm in ("randomized","passage_reader"):
        if arm=="randomized":
            fit = WORK/arm
            assert sha(fit/"final.safetensors") == load(WORK/"result.json")["fits"][arm]["sha256"]
            net = load_visual_writer(fit,DEVICE)
            feat,mem,_=physical_tables(bank,DEVICE)
            ink=net.ink(feat)
            imgs,ends=generate(net,ink,mem,net.write_key(ink[mem]),selected,qs,bank)
        else:
            assert sha(BASE/"mixed.safetensors")==load(BASE/"mixed_result.json")["weights_sha256"]
            net = OrderedPassageComposer().eval()
            load_model(net,str(BASE/"mixed.safetensors"))
            imgs,ends = zip(*(net.answer(as_grid(p),as_grid(q)) for p,q in zip(selected,qs,strict=True)),strict=True)
        path=out/(arm+".npz")
        image_archive(path,imgs,ends)
        ok = correct(imgs,ends)
        result[arm]=dict(exact=int(ok.sum()),both=int(ok[pairs].all(1).sum()),archive=sha(path),
                         matched_difference=clustered_delta(matched,ok,groups),
                         matched_pairs_difference=clustered_delta(matched[pairs].all(1),ok[pairs].all(1),groups[pairs[:,0]]))
    write_json(PUBLIC/"sentence_controls.json",dict(rows=len(idx),arms=result,program=sha(__file__),
               selector=sha(ROOT/"ilm/visual_lm/pictured_sentence.py")))


if __name__=="__main__":
    with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        compare()
