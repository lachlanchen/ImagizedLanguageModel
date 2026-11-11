#!/usr/bin/env python3
"""Register and run frozen-weight reader/writer/history diagnostics.

Reference grouping, native embeddings and next-span lengths are diagnostic
inputs. These outputs must never be scored as autonomous language answers.
"""
import argparse
from collections import defaultdict
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F

from ilm.image_edit.baseline import ROOT, save_json, sha
from ilm.visual_lm.compact_visual_span_weights import open_compact_archive
from ilm.visual_lm.visual_sequence_corpus import VisualSequenceCorpus
from scripts.infer_compact_visual_span import load_checkpoint, save_grid
from ilm.visual_lm.full_rank_visual_span import load_full_rank_checkpoint

DATA = ROOT / "data/visual_sequence_corpus_v1"
CODEC = ROOT / "artifacts/continuous_glyph_codec_v34_20260814/checkpoint_latest.pt"
LOCAL = ROOT.parent / "LocalLLM"
CHECKPOINTS = {
    "v1": ROOT / "artifacts/compact_visual_span_fit_v1/image_heads_step_0008172.pt",
    "wider": ROOT / "artifacts/compact_visual_span_full_rank_fit_v1/image_heads_step_0008172.pt",
}
SEED = 202611156
NAMESPACE = "compact-visual-span-failure-localization-v1"
SOURCES = (
    "scripts/diagnose_compact_visual_span.py",
    "ilm/visual_lm/compact_visual_span_model.py",
    "ilm/visual_lm/compact_visual_span_weights.py",
    "ilm/visual_lm/compact_visual_span_training.py",
    "ilm/visual_lm/full_rank_visual_span.py",
    "ilm/visual_lm/visual_sequence_corpus.py",
    "scripts/infer_compact_visual_span.py",
    "ilm/visual_lm/glyph_runtime_resources.py",
    "ilm/visual_lm/glyph_reader_monitor.py",
)
ARMS = ("visual_reference_8", "visual_reference_64", "native_reference_8",
        "shuffled_reference_8", "emitted_fixed_counts_8", "learned_prompt_emitted_8")


def load(path):
    return json.loads(Path(path).read_text())


def digest_tensor(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def order(value):
    return hashlib.sha256((NAMESPACE + "\0" + value).encode()).hexdigest()


def source_identity(committed=False):
    identity = {name: sha(ROOT / name) for name in SOURCES}
    if committed:
        for name, digest in identity.items():
            if hashlib.sha256(subprocess.check_output(["git", "show", "HEAD:" + name], cwd=ROOT)).hexdigest() != digest:
                raise ValueError("Commit diagnostic sources before registration: " + name)
    return identity


def selected_records(corpus):
    groups = {kind: defaultdict(list) for kind in ("qa", "book")}
    for shard_index, shard in enumerate(corpus.shards):
        name, = [name for name in shard["files"] if name.endswith("_sources.json")]
        if sha(DATA / name) != shard["files"][name]:
            raise ValueError("Corpus source index changed")
        rows = load(DATA / name)
        if len(rows) != shard["records"]:
            raise ValueError("Source index size differs")
        for local, row in enumerate(rows):
            groups[row["kind"]][row["group"]].append((corpus.offsets[shard_index] + local, row))
    result = []
    # Eight cases of each kind, taking one from each group in deterministic
    # rounds. The two development books therefore contribute four cases each.
    for kind in ("qa", "book"):
        queues = [sorted(rows, key=lambda row: order(row[1]["id"]))
                  for _, rows in sorted(groups[kind].items(), key=lambda pair: order(pair[0]))]
        chosen = []
        offset = 0
        while len(chosen) < 8:
            for queue in queues:
                if offset < len(queue):
                    chosen.append(queue[offset])
                    if len(chosen) == 8:
                        break
            offset += 1
            if offset > len(corpus):
                raise ValueError("Insufficient diagnostic cases")
        result.extend(chosen)
    return result


def affine_geometry(state, vectors):
    """Distance to the exact LayerNorm-centered feedback affine family.

    This relaxes all preceding image-network constraints. Its error is a lower
    bound for the fixed head on these source rows, not a language score.
    """
    e = state["image_heads"]["encoder"]
    w, gamma = e["feedback.1.weight"].double(), e["feedback.0.weight"].double()
    beta, bias = e["feedback.0.bias"].double(), e["feedback.1.bias"].double()
    scale, mean = e["feedback_scale"].double(), e["feedback_mean"].double()
    matrix = scale[:, None] * w * gamma[None]
    matrix = matrix - matrix.mean(dim=1, keepdim=True)
    offset = mean + scale * (w @ beta + bias)
    u, s, _ = torch.linalg.svd(matrix, full_matrices=False)
    tolerance = max(matrix.shape) * torch.finfo(torch.float64).eps * float(s[0])
    rank = int((s > tolerance).sum())
    q = u[:, :rank]
    result = []
    for case, target in vectors:
        target = target.double()
        centered = target - offset
        residual = centered - (centered @ q) @ q.T
        relative = residual.square().sum(-1) / target.square().sum(-1).clamp_min(1e-8)
        result.append({"case": case, "rows": len(target), "minimum_relative_error_sum": float(relative.sum()),
                       "minimum_relative_error_mean": float(relative.mean())})
    return {"feedback_shape": list(w.shape), "centered_affine_rank": rank,
            "tolerance_float64": tolerance, "smallest_singular_value": float(s[-1]),
            "orthogonality_error": float((q.T @ q - torch.eye(rank, dtype=torch.float64)).abs().max()),
            "scope": "Fixed trained feedback head, exact-arithmetic affine relaxation, uniform selected source positions; not the full training objective's batch weighting or a language bound",
            "cases": result}


def prepare(output):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or output.exists():
        raise ValueError("Prepare on CPU with a new output directory")
    torch.set_num_threads(2)
    began = time.perf_counter()
    code = source_identity(committed=True)
    report = load(DATA / "report.json")
    verification = load(DATA / "verification.json")
    if verification["report_sha256"] != sha(DATA / "report.json") or verification["cross_partition_exact_input_or_source_group_overlap"] != 0:
        raise ValueError("Use the verified group-disjoint corpus")
    cases, rows = [], []
    for split in ("train", "development"):
        corpus = VisualSequenceCorpus(DATA, split=split)
        for index, origin in selected_records(corpus):
            row, alignment = corpus[index], corpus.offline_teacher_alignment(index)
            ends, unit_ends = row["span_glyph_ends"], alignment["span_unit_ends"]
            prefix = row["prompt_spans"] if origin["kind"] == "qa" else max(1, len(ends) // 2)
            stop = min(len(ends), prefix + 8)
            if not 0 < prefix < stop or int(ends[-1]) != len(row["ink"]):
                raise ValueError("Diagnostic prefix/target alignment invalid")
            if origin["kind"] == "qa" and int(ends[prefix - 1]) != row["prompt_glyphs"]:
                raise ValueError("QA prompt boundary differs from supplied source")
            counts = torch.diff(unit_ends, prepend=unit_ends.new_zeros(1))
            if bool(((counts < 1) | (counts > 3)).any()):
                raise ValueError("Reference update counts exceed model capacity")
            key = f"{split}_{origin['kind']}_{len(cases):02d}"
            cases.append(dict(id=key, split=split, kind=origin["kind"], record_index=index,
                              source_id=origin["id"], source_group=origin["group"],
                              source_text_sha256=origin["source_text_sha256"],
                              ink_sha256=digest_tensor(row["ink"]),
                              unit_ids_sha256=digest_tensor(alignment["unit_ids"]),
                              glyph_ends_sha256=digest_tensor(ends), unit_ends_sha256=digest_tensor(unit_ends),
                              prefix_spans=prefix, stop_span=stop, target_spans=stop-prefix,
                              target_glyphs=int(ends[stop-1]-ends[prefix-1])))
            rows.append(alignment)
    if len(cases) != 32:
        raise ValueError("The fixed diagnostic needs32 records")
    compact = open_compact_archive(LOCAL)
    from ilm.visual_lm.offline_visual_sequence_teacher import open_archive
    teacher = open_archive(LOCAL)
    compact_tokens = compact.reader.fields["tokenizer.ggml.tokens"].contents()
    teacher_tokens = teacher.field("tokenizer.ggml.tokens")
    training_units = load(ROOT / "artifacts/visual_sequence_teacher_v1/preparation.json")["training_native_units"]
    all_units = sorted(set(training_units).union(*(set(row["unit_ids"].tolist()) for row in rows)))
    different = [i for i in all_units if i >= len(compact_tokens) or compact_tokens[i] != teacher_tokens[i]]
    if different:
        raise ValueError("Native source IDs are not interchangeable: " + str(different[:12]))
    stored = compact.omitted["token_embd.weight"]
    unique = sorted(set().union(*(set(row["unit_ids"].tolist()) for row in rows)))
    table = {}
    for begin in range(0, len(unique), 512):
        ids = unique[begin:begin+512]
        values = compact.gguf.dequantize(stored.data[ids], stored.tensor_type)
        for unit, vector in zip(ids, values):
            table[unit] = torch.from_numpy(np.array(vector, dtype=np.float32, copy=True)).to(torch.bfloat16)
    vectors = {case["id"]: torch.stack([table[int(i)] for i in row["unit_ids"]]) for case, row in zip(cases, rows)}
    geometry, checkpoints = {}, {}
    for name, path in CHECKPOINTS.items():
        state = torch.load(path, weights_only=True, map_location="cpu")
        if state["updates"] != 8172 or state["identity"]["codec_sha256"] != sha(CODEC):
            raise ValueError("Use completed matched fits and codec")
        geometry[name] = affine_geometry(state, list(vectors.items()))
        checkpoints[name] = {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}
        del state
    output.mkdir(parents=True)
    torch.save(vectors, output / "source_vectors.pt")
    save_json(output / "feedback_geometry.json", geometry)
    request = dict(schema=NAMESPACE, sources_sha256=code, checkpoints=checkpoints,
                   corpus_report_sha256=sha(DATA / "report.json"), corpus_verification_sha256=sha(DATA / "verification.json"),
                   codec_sha256=sha(CODEC), cases=cases, native_vectors_sha256=sha(output / "source_vectors.pt"),
                   geometry_sha256=sha(output / "feedback_geometry.json"),
                   vocabulary_row_audit=dict(compact_vocabulary=len(compact_tokens), teacher_vocabulary=len(teacher_tokens),
                                             training_units=len(training_units), audited_units=len(all_units), mismatches=len(different)),
                   seed=SEED, arms=list(ARMS), reference_lengths_supplied=True, maximum_target_spans_per_record=8,
                   integration_steps=[8,64], maximum_seconds=1800, pressure_wait_allowance_seconds=300,
                   maximum_velocity_calls=2*104*sum(case['target_spans'] for case in cases),
                   maximum_body_calls=2*sum(5+2*case['target_spans'] for case in cases),
                   minimum_entry_host_ram_gib=48, minimum_runtime_host_ram_gib=24,
                   maximum_allocated_vram_gib=22, minimum_free_gpu_gib=20,
                   primary_measurements=["foreground_f1", "exact_glyphs", "relative_latent_mse", "feedback_relative_mse",
                                         "feedback_affine_floor", "update_count_accuracy", "boundary_f1", "context_cosine"],
                   interpretation="Reference-supplied reconstruction and representation diagnostics; not autonomous language accuracy",
                   first_state_relative_rms_tolerance=.02, model_calls_during_preparation=0,
                   cuda_initialized=torch.cuda.is_initialized(), preparation_seconds=time.perf_counter()-began)
    save_json(output / "registration.json", request)
    print(json.dumps({"prepared":True,"cases":len(cases),"target_spans":sum(c['target_spans'] for c in cases),
                      "vocabulary":request["vocabulary_row_audit"],"geometry":{k:{a:b for a,b in v.items() if a!='cases'} for k,v in geometry.items()},
                      "registration_sha256":sha(output/'registration.json')},indent=2),flush=True)


def image_metrics(predicted, gold):
    p, g = predicted.bool(), gold.bool()
    tp, fp, fn = int((p & g).sum()), int((p & ~g).sum()), int((~p & g).sum())
    exact = int((p == g).flatten(1).all(1).sum())
    return dict(glyphs=len(g), exact_glyphs=exact, tp=tp, fp=fp, fn=fn,
                foreground_f1=2*tp/max(2*tp+fp+fn,1), pixel_equal=float((p == g).float().mean()))


def relative_mse(a,b):
    return float(((a.float()-b.float()).square().sum(-1)/b.float().square().sum(-1).clamp_min(1e-8)).mean())


def context_cosine(a,b):
    return float(F.cosine_similarity(a.float().reshape(-1),b.float().reshape(-1),dim=0))


def noise_seed(case, span):
    return SEED + int(order(case + ':' + str(span))[:8],16)


@torch.inference_mode()
def diagnose_model(model, name, cases, vectors, output, monitor, device, request):
    cached = []
    corpora = {split: VisualSequenceCorpus(DATA, split=split) for split in ("train", "development")}
    for case in cases:
        monitor.position = {"model":name,"phase":"reference_contexts","case":case["id"]};monitor.guard()
        corpus = corpora[case["split"]]
        row, aligned = corpus[case["record_index"]], corpus.offline_teacher_alignment(case["record_index"])
        for actual, key in ((row["ink"],"ink_sha256"),(aligned["unit_ids"],"unit_ids_sha256"),
                            (row["span_glyph_ends"],"glyph_ends_sha256"),(aligned["span_unit_ends"],"unit_ends_sha256")):
            if digest_tensor(actual) != case[key]:raise ValueError("Registered case changed")
        ends = row["span_glyph_ends"].tolist(); starts = [0,*ends[:-1]]
        spans = [s.to(device) for s in row["ink"].split([e-s for s,e in zip(starts,ends)])]
        counts = torch.diff(aligned["span_unit_ends"],prepend=aligned["span_unit_ends"].new_zeros(1)).to(device).long()
        observation = model.observe(spans,counts,[len(spans)])
        valid = torch.arange(3,device=device)[None] < counts[:,None]
        native = vectors[case["id"]].to(device)
        predicted = observation.feedback_vectors[valid]
        native_states = model._body(native[None],use_cache=False).last_hidden_state[0,aligned["span_unit_ends"].to(device).long()-1]
        prefix = case["prefix_spans"]
        prefix_ink = row["ink"][:ends[prefix-1]].to(device)
        learned, grouping = model.read_prompt(prefix_ink)
        learned_state = model._body(learned,use_cache=False).last_hidden_state[0,-1]
        boundaries = model.encoder.boundary_logits(model.glyph_latents(prefix_ink)) >= 0
        gold_boundary = torch.zeros_like(boundaries);gold_boundary[torch.tensor(ends[:prefix],device=device)-1] = True
        tp=int((boundaries & gold_boundary).sum());fp=int((boundaries & ~gold_boundary).sum());fn=int((~boundaries & gold_boundary).sum())
        reader = dict(feedback_relative_mse=relative_mse(predicted,native),update_rows=len(native),
                      correct_counts=int((observation.feedback_count_logits.argmax(-1)==counts-1).sum()),
                      spans=len(counts),boundary_tp=tp,boundary_fp=fp,boundary_fn=fn,
                      boundary_f1=2*tp/max(2*tp+fp+fn,1),
                      learned_prompt_spans=len(grouping),reference_prompt_spans=prefix,
                      learned_prompt_updates=learned.shape[1],reference_prompt_updates=int(counts[:prefix].sum()),
                      grouping_exact=grouping==[len(s) for s in spans[:prefix]],
                      image_vs_native_context_cosine=context_cosine(observation.contexts[prefix-1],native_states[prefix-1]),
                      learned_vs_reference_context_cosine=context_cosine(learned_state,observation.contexts[prefix-1]))
        cached.append(dict(case=case,spans=[s.cpu() for s in spans],counts=counts.cpu(),
                           latents=[v.cpu() for v in observation.glyph_latents],
                           states=observation.contexts.cpu(),native_states=native_states.cpu(),reader=reader))
        del spans,observation,native_states,native,predicted,learned,learned_state,prefix_ink
    # Wrong contexts come from the next complete record's first target state,
    # without consulting any diagnostic error or output quality.
    records=[]
    for ci, data in enumerate(cached):
        case=data['case'];key=case['id'];prefix=case['prefix_spans'];stop=case['stop_span']
        monitor.status('diagnosing_case',model=name,case=key)
        directory=output/name/key;directory.mkdir(parents=True,exist_ok=False)
        targets=data['spans'][prefix:stop];gold=torch.cat(targets)
        save_grid(gold,directory/'reference.png')
        save_grid(torch.cat(data['spans'][:prefix]),directory/'input.png')
        target_latent=torch.cat(data['latents'][prefix:stop]).to(device)
        reconstruction=(model.codec.decode(target_latent)<0).to(torch.uint8).cpu()
        save_grid(reconstruction,directory/'codec.png')
        codec_metrics=image_metrics(reconstruction,gold)
        other=cached[(ci+1)%len(cached)];wrong=other['states'][other['case']['prefix_spans']-1].to(device)[None]
        arms={};first_context_errors={}
        for arm in ARMS:
            monitor.position={'model':name,'phase':arm,'case':key};monitor.guard()
            current=None
            if arm in ('emitted_fixed_counts_8','learned_prompt_emitted_8'):
                if arm=='learned_prompt_emitted_8':
                    inputs,_=model.read_prompt(torch.cat(data['spans'][:prefix]).to(device))
                else:
                    spans=[s.to(device) for s in data['spans'][:prefix]]
                    v,_=model.encode_ink_spans(spans)
                    mask=torch.arange(3,device=device)[None]<data['counts'][:prefix,None].to(device)
                    inputs=v[mask][None]
                current=model._body(inputs,use_cache=True)
                if arm=='emitted_fixed_counts_8':
                    err=relative_mse(current.last_hidden_state[:,-1],data['states'][prefix-1].to(device)[None])
                    first_context_errors[arm]=err
                    if err**.5>request['first_state_relative_rms_tolerance']:
                        raise ValueError('Reference prefix state does not match causal target indexing')
            outputs=[];latents=[];trace=[];length_correct=0
            for span_index in range(prefix,stop):
                monitor.guard()
                if current is not None:state=current.last_hidden_state[:,-1]
                elif arm=='native_reference_8':state=data['native_states'][span_index-1].to(device)[None]
                elif arm=='shuffled_reference_8':state=wrong
                else:state=data['states'][span_index-1].to(device)[None]
                count=len(data['spans'][span_index]);steps=64 if arm=='visual_reference_64' else 8
                length_correct+=int(int(model.writer.length(state.float()).argmax(-1)[0])==count)
                seed=noise_seed(key,span_index)
                generator=torch.Generator(device=device).manual_seed(seed)
                # Capture and reset the actual sampler's initial noise for all
                # arms, rather than assuming generator construction is enough.
                noise=torch.randn((1,count,768),device=device,generator=generator)
                noise_sha=digest_tensor(noise)
                generator.manual_seed(seed)
                latent=model.writer.sample(state,count,steps,generator)
                ink=(model.codec.decode(latent)<0).to(torch.uint8)
                event=dict(span=span_index,glyphs=count,noise_sha256=noise_sha,ink_sha256=digest_tensor(ink),
                           context_vs_reference_cosine=context_cosine(state,data['states'][span_index-1].to(device)[None]))
                if current is not None:
                    v,counts=model.encode_ink_spans([ink])
                    updates=int(data['counts'][span_index]) if arm=='emitted_fixed_counts_8' else int(counts.argmax(-1)[0])+1
                    current=model._body(v[:,:updates],past_key_values=current.past_key_values,use_cache=True)
                    event.update(emitted_image_feedback=True,continuous_updates=updates,feedback_vectors_sha256=digest_tensor(v[:,:updates]))
                outputs.append(ink.cpu());latents.append(latent.cpu());trace.append(event)
            pixels=torch.cat(outputs);latent=torch.cat(latents)
            save_grid(pixels,directory/(arm+'.png'))
            np.savez_compressed(directory/(arm+'.npz'),ink=pixels.numpy(),latent=latent.numpy())
            arms[arm]={**image_metrics(pixels,gold), 'relative_latent_mse':relative_mse(latent,target_latent.cpu()),
                       'target_spans':stop-prefix,'predicted_length_correct':length_correct,'trace':trace,
                       'png_sha256':sha(directory/(arm+'.png')),'raw_sha256':sha(directory/(arm+'.npz'))}
            del current
        for j in range(stop-prefix):
            if len({arms[arm]['trace'][j]['noise_sha256'] for arm in ARMS})!=1:
                raise ValueError('Intervention arms did not receive exactly matched noise')
        record=dict(**case,reader=data['reader'],codec=codec_metrics,arms=arms,
                    fixed_prefix_causal_context_relative_mse=first_context_errors)
        save_json(directory/'report.json',record);records.append(record)
        print(json.dumps({'model':name,'case':key,'reader_mse':data['reader']['feedback_relative_mse'],
                          'codec_f1':codec_metrics['foreground_f1'],
                          'arms_f1':{a:r['foreground_f1'] for a,r in arms.items()}}),flush=True)
    return records


def run(output):
    request=load(output/'registration.json')
    if request['sources_sha256']!=source_identity(committed=True):raise ValueError('Diagnostic sources changed')
    for path,digest in ((DATA/'report.json',request['corpus_report_sha256']),
                        (DATA/'verification.json',request['corpus_verification_sha256']),
                        (CODEC,request['codec_sha256']),
                        (output/'source_vectors.pt',request['native_vectors_sha256']),
                        (output/'feedback_geometry.json',request['geometry_sha256'])):
        if sha(path)!=digest:raise ValueError('Registered diagnostic input changed: '+str(path))
    for row in request['checkpoints'].values():
        if sha(ROOT/row['path'])!=row['sha256']:raise ValueError('Checkpoint changed')
    execution=output/'execution';execution.mkdir(exist_ok=False)
    from ilm.visual_lm.glyph_reader_monitor import GlyphReaderMonitor
    from ilm.visual_lm.glyph_runtime_resources import full_reader_gpu
    monitor=GlyphReaderMonitor(execution,request['maximum_seconds'])
    monitor.pressure_wait_allowance_seconds=request['pressure_wait_allowance_seconds']
    monitor.status('admitting')
    vectors=torch.load(output/'source_vectors.pt',weights_only=True,map_location='cpu')
    try:
        with full_reader_gpu() as (device,admission):
            monitor.admission=admission;torch.set_num_threads(2)
            summaries=[]
            for name in ('v1','wider'):
                monitor.guard();monitor.status('loading_frozen_model',model=name)
                loader=load_checkpoint if name=='v1' else load_full_rank_checkpoint
                model,updates,provenance=loader(ROOT/request['checkpoints'][name]['path'],LOCAL,CODEC,device)
                if updates!=8172 or any(p.requires_grad for p in model.parameters()):raise ValueError('Frozen final model required')
                records=diagnose_model(model,name,request['cases'],vectors,execution,monitor,device,request)
                save_json(execution/(name+'_records.json'),records)
                summaries.append({'model':name,'records':len(records),'provenance':provenance})
                del model;gc.collect();torch.cuda.empty_cache()
            save_json(execution/'report.json',dict(registration_sha256=sha(output/'registration.json'),
                      models=summaries,reference_supplied_diagnostics=True,autonomous_language_qualified=False,
                      optimizer_updates=0,**monitor.receipt()))
            monitor.status('complete')
            terminal=load(execution/'status.json');terminal['pid']=None
            save_json(execution/'status.json',terminal)
    except BaseException as error:
        monitor.status('failed',error=repr(error));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('prepare','run'))
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/compact_visual_span_diagnosis_v1')
    args=parser.parse_args()
    if args.phase=='prepare':prepare(args.output)
    else:run(args.output)


if __name__=='__main__':main()
