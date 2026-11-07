#!/usr/bin/env python3
"""Prepare real image supervision and qualify contextual-span differentiation."""
import argparse
import gc
import json
from pathlib import Path
import time
import traceback

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from ilm.visual_lm.contextual_pixel_learning import LearningPixelReader, fixed_grid_pool, pooling_matrix
from ilm.visual_lm.contextual_pixel_spans import units, copied, matches
from ilm.visual_lm.causal_visual_prefix import fold_writing_prompt
from ilm.visual_lm.full_text_renderer import CompleteLineRenderer
from ilm.visual_lm.semantic_image_memory import native_budget_image
from scripts.cache_semantic_images_v72 import ROOT, CONTROLS, CONTROL_SHA, sha, atomic_json, committed_identity
from scripts.run_contextual_pixel_spans import source_rows, array, read
from scripts.probe_native_unified_source import gpu_slot
from scripts.train_clean_corpus_v75 import setup

PREVIOUS = ROOT / "artifacts/contextual_pixel_spans_v1"
HEAD_SHA = "85369b1c1547f6fd02e5cc7ea601e202bb03ed8360e6e4964e94b5beec476a91"
SEED = 2026110705
SOURCES = ("scripts/check_contextual_pixel_learning.py", "ilm/visual_lm/contextual_pixel_learning.py",
    "references/contextual_pixel_learning_protocol.md", "ilm/visual_lm/contextual_pixel_encoder.py",
    "ilm/visual_lm/contextual_pixel_spans.py", "ilm/visual_lm/grounded_image_actor.py",
    "ilm/visual_lm/grounded_image_weights.py", "ilm/visual_lm/image_context_interface.py",
    "ilm/visual_lm/causal_visual_prefix.py", "ilm/visual_lm/semantic_image_memory.py",
    "ilm/visual_lm/full_text_renderer.py", "scripts/run_contextual_pixel_spans.py")


def event(root, stage, **values):
    import os
    record = dict(stage=stage, pid=os.getpid())
    record.update(values)
    atomic_json(root / "status.json", record)
    print(json.dumps(record), flush=True)


def physical(pixels):
    return torch.from_numpy(pixels.copy()).float().div(127.5).sub(1)[None, None]


def prepare(args):
    folder = args.run / "prepared"; folder.mkdir(parents=True, exist_ok=False)
    identity = committed_identity(SOURCES)
    renderer = CompleteLineRenderer()
    rows, refs, rejected = [], [], []
    old_rows = read(PREVIOUS / "prepared/inputs.json")
    old_refs = {r["id"]:r for r in read(PREVIOUS / "prepared/references.json")}
    development = [r for r in old_rows if r["split"] == "development" and r["task"] == "qa"]
    dev_articles, dev_pairs = {r["article"] for r in development}, {r["pair"] for r in development}
    sources = sorted(source_rows("train"), key=lambda r:r["id"])
    assert len(sources) == 3211
    if {r["article_title"] for r in sources} & dev_articles or {r["context_hash"] for r in sources} & dev_pairs:
        raise ValueError("training/development source overlap")
    for row in sources:
        try:
            pixels, meta = renderer.render(row["prompt"], maximum_blocks=240)
            layout, _ = renderer._layout(row["prompt"])
            u = units(pixels)
            targets = []
            def x(index):
                return meta["draw_x"] + layout.index_to_pos(len(row["prompt"][:index].encode("utf8"))).x / renderer.Pango.SCALE
            for ref in row["references"]:
                begin, end = 3 + ref["start"], 3 + ref["end"]
                if row["prompt"][begin:end] != ref["text"]:
                    raise ValueError("source reference offset changed")
                lo, hi = x(begin), x(end)
                selected = np.flatnonzero((u[:, 1] > lo) & (u[:, 0] < hi))
                if not len(selected) or u[selected[0], 0] < lo - 1 or u[selected[-1], 1] > hi + 1 or len(selected)>128:
                    raise ValueError("reference cuts an indivisible ink run")
                targets.append(copied(pixels, u, int(selected[0]), len(selected)))
            matches(pixels, u, targets)
            shape = list(fold_writing_prompt(physical(pixels)).pixels.shape[-2:])
        except (ValueError, OverflowError) as error:
            rejected.append(dict(id=row["id"], reason=str(error)))
            continue
        index = len(rows)
        path = folder / f"inputs/{index:04d}.png"; path.parent.mkdir(exist_ok=True)
        Image.fromarray(pixels).save(path)
        rows.append(dict(id=row["id"], task="qa", language="zh", split="train", article=row["article_title"],
            pair=row["context_hash"], input=str(path.relative_to(ROOT)), input_sha256=sha(path),
            folded_shape=shape, ink_units=len(u)))
        files = []
        for i, target in enumerate(targets):
            p = folder / f"references/{index:04d}_{i}.png"; p.parent.mkdir(exist_ok=True)
            Image.fromarray(target).save(p); files.append(dict(file=str(p.relative_to(ROOT)), sha256=sha(p)))
        refs.append(dict(id=row["id"], targets=files))
        if len(rows)%400==0: print("Prepared training QA",len(rows),flush=True)
    for row in old_rows:
        if row["split"] != "train" or row["task"] != "copy": continue
        p = PREVIOUS / "prepared" / row["input"]
        if sha(p) != row["input_sha256"]: raise ValueError("copy input changed")
        pixels = array(p)
        rows.append(dict(**{k:v for k,v in row.items() if k not in ("input","input_sha256")},
            input=str(p.relative_to(ROOT)), input_sha256=sha(p), ink_units=len(units(pixels)),
            folded_shape=list(fold_writing_prompt(physical(pixels)).pixels.shape[-2:])))
        files = []
        for target in old_refs[row["id"]]["targets"]:
            p = PREVIOUS / "prepared" / target["file"]
            if sha(p) != target["sha256"]: raise ValueError("copy target changed")
            files.append(dict(file=str(p.relative_to(ROOT)), sha256=sha(p)))
        refs.append(dict(id=row["id"], targets=files))
    if len({r["id"] for r in rows}) != len(rows): raise ValueError("duplicate training ID")
    controls = torch.load(CONTROLS, map_location="cpu", weights_only=True)
    budgets = {}
    for shape in sorted({tuple(r["folded_shape"]) for r in rows}):
        transformed = native_budget_image(torch.ones(1,1,*shape))
        budgets[shape] = transformed.shape[-2] * transformed.shape[-1] // 1024 + len(controls["before"]) + len(controls["after"])
    for row in rows: row["context_positions"] = budgets[tuple(row["folded_shape"])]
    previous_ids = {r["id"] for r in old_rows if r["split"] == "train"}
    new = [r["id"] for r in rows if r["task"] == "qa" and r["id"] not in previous_ids]
    resource = dict(largest_sequence=max(rows, key=lambda r:r["context_positions"])["id"],
                    most_ink_units=max(rows, key=lambda r:r["ink_units"])["id"])
    atomic_json(folder / "inputs.json", rows); atomic_json(folder / "references.json", refs)
    atomic_json(folder / "report.json", dict(identity=identity, inputs_sha256=sha(folder/"inputs.json"),
        references_sha256=sha(folder/"references.json"), training_qa=sum(r["task"]=="qa" for r in rows),
        training_copies=sum(r["task"]=="copy" for r in rows), source_training_rows=len(sources),
        rejected=rejected, renderer=renderer.provenance, gradient_case_ids=new[:2], resource_case_ids=resource,
        development_inputs_sha256=sha(PREVIOUS/"prepared/inputs.json"), source_articles_disjoint=True,
        source_paragraphs_disjoint=True, outcomes_used_to_select_examples=False, general_ilm_goal_complete=False))
    event(args.run,"prepared",training_images=len(rows),rejected=len(rejected))


def check_pool():
    torch.manual_seed(SEED)
    field = torch.randn(1,3,4,5,dtype=torch.float64,requires_grad=True)
    grid = torch.rand(1,4,9,2,dtype=torch.float64)*3-1.5
    reference = F.grid_sample(field,grid,align_corners=False,padding_mode="border").mean(-1).transpose(1,2)[0]
    actual = fixed_grid_pool(field,grid)
    gradient = torch.randn_like(actual)
    a, = torch.autograd.grad(actual,field,gradient)
    b, = torch.autograd.grad(reference,field,gradient)
    if not torch.equal(actual,reference) or not torch.allclose(a,b,atol=1e-12,rtol=1e-12):
        raise ValueError("fixed-grid adjoint differs from CPU reference")
    if not torch.autograd.gradcheck(lambda f:fixed_grid_pool(f,grid),(field,),fast_mode=True):
        raise ValueError("fixed-grid adjoint fails finite differences")
    matrix = pooling_matrix(grid,4,5)
    if not torch.allclose(matrix.sum(1),torch.ones(4,dtype=matrix.dtype),atol=1e-14,rtol=0):
        raise ValueError("sampling weights do not conserve constants")
    return dict(native_forward_bit_exact=True,cpu_adjoint_max_error=float((a-b).abs().max()),
                finite_difference_pass=True,border_samples_present=bool((grid.abs()>1).any()))


def cpu_tree(value):
    if isinstance(value,torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value,dict): return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list): return [cpu_tree(v) for v in value]
    if isinstance(value,tuple): return tuple(cpu_tree(v) for v in value)
    return value


def exact(a,b):
    if isinstance(a,torch.Tensor): return isinstance(b,torch.Tensor) and a.dtype==b.dtype and torch.equal(a,b)
    if isinstance(a,dict): return a.keys()==b.keys() and all(exact(v,b[k]) for k,v in a.items())
    if isinstance(a,(list,tuple)): return len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
    return a==b


def preflight(args):
    identity = committed_identity(SOURCES)
    folder = args.run / "preflight"; folder.mkdir(exist_ok=False)
    prep = read(args.run/"prepared/report.json")
    if prep["identity"]["sources"] != identity["sources"]: raise ValueError("preparation code changed")
    rows = {r["id"]:r for r in read(args.run/"prepared/inputs.json")}
    refs = {r["id"]:r for r in read(args.run/"prepared/references.json")}
    if sha(args.run/"prepared/inputs.json")!=prep["inputs_sha256"] or sha(args.run/"prepared/references.json")!=prep["references_sha256"]:
        raise ValueError("prepared indices changed")
    pool = check_pool(); atomic_json(folder/"pool_adjoint.json",pool)
    with gpu_slot():
        device=setup(); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        local=ROOT.parent/"LocalLLM/.local"; event(args.run,"loading_native_context")
        began=time.monotonic()
        model=LearningPixelReader(checkpoint=PREVIOUS/"training/checkpoint.pt",checkpoint_sha256=HEAD_SHA,
            controls=CONTROLS,controls_sha256=CONTROL_SHA,ollama_root=local/"models/ollama",
            gguf_source=local/"tools/llama.cpp/gguf-py",device=device)
        setup_seconds=time.monotonic()-began
        old=read(PREVIOUS/"prepared/inputs.json")
        cases=[r for r in old if r["split"]=="development" and r["task"]=="qa" and r["condition"]=="normal"][:2]
        cases += [r for r in old if r["split"]=="development" and r["task"]=="copy"][:2]
        cache={r["id"]:r for r in read(PREVIOUS/"features/report.json")["records"]}
        calibrated=[]
        for row in cases:
            p=PREVIOUS/"prepared"/row["input"]; c=PREVIOUS/"features"/cache[row["id"]]["file"]
            if sha(p)!=row["input_sha256"] or sha(c)!=cache[row["id"]]["sha256"]: raise ValueError("calibration changed")
            expected=torch.load(c,map_location="cpu",weights_only=True)
            with torch.no_grad(): values=model.encode(array(p))
            for name in ("visual","memory","geometry","source_units"):
                if not torch.equal(values[name].cpu(),expected[name]): raise ValueError("native forward changed: "+name)
            if values["metadata"]!=expected["metadata"]: raise ValueError("native geometry changed")
            calibrated.append(dict(id=row["id"],exact=True))
            del values
        model.train()
        # Checkpointed training forward must also preserve the source interface.
        row=cases[0]; p=PREVIOUS/"prepared"/row["input"]
        expected=torch.load(PREVIOUS/"features"/cache[row["id"]]["file"],map_location="cpu",weights_only=True)
        values=model.encode(array(p))
        if any(not torch.equal(values[n].detach().cpu(),expected[n]) for n in ("visual","memory","geometry","source_units")):
            raise ValueError("checkpointed training forward differs at zero adapters")
        del values,expected; gc.collect(); torch.cuda.empty_cache()
        optimizer=model.optimizer(); params=model.trained_parameters()
        if {id(p) for group in optimizer.param_groups for p in group["params"]}!={id(p) for p in params.values()}:
            raise ValueError("Adam parameter set differs")

        def update(key, require_down=False, commit=True):
            row=rows[key]; p=ROOT/row["input"]
            if sha(p)!=row["input_sha256"]: raise ValueError("training input changed")
            image=array(p); targets=[]
            for target in refs[key]["targets"]:
                p=ROOT/target["file"]
                if sha(p)!=target["sha256"]: raise ValueError("training target changed")
                targets.append(array(p))
            labels=matches(image,units(image),targets)
            optimizer.zero_grad(set_to_none=True); started=time.monotonic()
            logp,values=model(image)
            loss=model.head.loss(logp[None],[labels])
            if not torch.isfinite(loss): raise FloatingPointError("nonfinite image-span loss")
            loss.backward()
            gradients={n:float(p.grad.norm()) for n,p in params.items() if p.grad is not None}
            if len(gradients)!=len(params) or any(not torch.isfinite(p.grad).all() for p in params.values()):
                raise FloatingPointError("missing or nonfinite trained gradients")
            for name,norm in gradients.items():
                if (name.endswith(".up") or (require_down and name.endswith(".down"))) and norm<=0:
                    raise ValueError("zero contextual adapter gradient: "+name)
            if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
                raise ValueError("frozen parameter has a gradient")
            norm=float(torch.nn.utils.clip_grad_norm_(list(params.values()),1.,error_if_nonfinite=True))
            if commit: optimizer.step()
            result=dict(id=key,loss=float(loss.detach()),gradient_norm=norm,gradient_norms=gradients,
                context_positions=values["metadata"]["sequence_length"],ink_units=len(values["visual"]))
            del logp,values,loss
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            return result,time.monotonic()-started

        event(args.run,"real_image_backward")
        torch.cuda.reset_peak_memory_stats(); began=time.monotonic()
        first,t1=update(prep["gradient_case_ids"][0])
        second,t2=update(prep["gradient_case_ids"][1],require_down=True)
        resource=[]
        for key in dict.fromkeys(prep["resource_case_ids"].values()):
            event(args.run,"resource_backward",id=key)
            result,seconds=update(key,require_down=True,commit=False)
            resource.append(dict(**result,seconds=seconds))
        def snapshot():
            return dict(model=model.trained_state(),optimizer=cpu_tree(optimizer.state_dict()),
                cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(device))
        saved=snapshot(); torch.save(saved,folder/"resume_check.pt")
        continuation,tc=update(prep["gradient_case_ids"][0],require_down=True)
        expected=snapshot()
        saved=torch.load(folder/"resume_check.pt",map_location="cpu",weights_only=True)
        model.restore_trained_state(saved["model"]); optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["cpu_rng"]); torch.cuda.set_rng_state(saved["cuda_rng"],device)
        replay,tr=update(prep["gradient_case_ids"][0],require_down=True)
        resumed=snapshot()
        if continuation!=replay or not exact(expected,resumed): raise ValueError("optimizer/RNG replay differs")
        peak=torch.cuda.max_memory_allocated()/1024**3
        if peak>=23: raise RuntimeError("full backward exceeds registered 23 GiB allocation limit")
        report=dict(identity=identity,preparation_sha256=sha(args.run/"prepared/report.json"),passed=True,
            pool_adjoint=pool,native_source_calibration=calibrated,checkpointed_forward_exact=True,
            full_runtime_parameters=sum(p.numel() for p in model.parameters()),
            trained_parameters=sum(p.numel() for p in params.values()),
            head_parameters=sum(p.numel() for p in model.head.parameters()),
            trained_tensor_count=len(params),setup_seconds=setup_seconds,
            first_updates=[dict(**first,seconds=t1),dict(**second,seconds=t2)],resource_backward=resource,
            exact_optimizer_rng_resume=True,resume_update_seconds=[tc,tr],
            preflight_training_seconds=time.monotonic()-began,peak_allocated_vram_gib=peak,
            peak_reserved_vram_gib=torch.cuda.max_memory_reserved()/1024**3,
            frozen_parameter_gradients_absent=True,production_training_started=False,
            source_pretraining_retained=True,language_quality_claim=False,free_generation_demonstrated=False,
            general_ilm_goal_complete=False)
        atomic_json(folder/"report.json",report)
        event(args.run,"preflight_complete",pid=None,peak_allocated_vram_gib=peak)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--stage",choices=("prepare","preflight"),required=True)
    args=parser.parse_args(); args.run=args.run.absolute(); args.run.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    try: {"prepare":prepare,"preflight":preflight}[args.stage](args)
    except BaseException as error:
        atomic_json(args.run/"failure.json",dict(error=repr(error),traceback=traceback.format_exc(),pid=None))
        event(args.run,"failed",pid=None,error=repr(error))
        raise
