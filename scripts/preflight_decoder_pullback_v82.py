#!/usr/bin/env python3
"""Check the proposed writing metric on the actual codec and retained errors."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import resource
import time

import torch
from torch.nn import functional as F

from ilm.visual_lm.codec_block_writer import load_qualified_writing_codec
from ilm.visual_lm.clean_image_actor import tensor_digest
from ilm.visual_lm.decoder_pullback import decoder_pullback_error, normalized_decoder_logits
from scripts.audit_native_neural_transcripts_v80 import write
from scripts.cache_semantic_images_v72 import ROOT, CODEC, sha, committed_identity
from scripts.probe_native_corpus_fit_v80 import cases_and_data, targets
from scripts.train_clean_corpus_v75 import setup

PLAN = "references/decoder_pullback_v82_preflight_protocol.md"
SOURCES = ("scripts/preflight_decoder_pullback_v82.py", PLAN, "ilm/visual_lm/decoder_pullback.py",
    "ilm/visual_lm/continuous_glyph_codec.py", "ilm/visual_lm/codec_block_writer.py",
    "scripts/probe_native_corpus_fit_v80.py", "scripts/train_native_chinese_v80.py")


def error(actual, reference):
    difference = actual.double()-reference.double()
    return dict(relative_l2=float(difference.norm()/reference.double().norm().clamp_min(1e-30)),
        maximum_absolute=float(difference.abs().max()))


def synchronize(device):
    if device.type == "cuda":torch.cuda.synchronize(device)


def inputs():
    run = ROOT/"artifacts/adaptive_readout_v81eval_20261107"
    verified = json.loads((run/"verification.json").read_text())
    if (verified["report_sha256"] != sha(run/"report.json")
            or not verified["all_64_outputs_independently_scored"]
            or verified["training_fit_repair_pass"]):
        raise ValueError("the completed authenticated failed repair is required")
    cases, data, indices = cases_and_data()
    predictions, codes, bindings = [], [], []
    for i, case in enumerate(cases[:4]):
        folder = run/"adapted/predictions"/f"{i:02}"
        header = json.loads((folder/"header.json").read_text())
        if header["source"] != case["source"]:raise ValueError("fixed case changed")
        for name, digest in header["files_sha256"].items():
            if sha(folder/name) != digest:raise ValueError("retained evidence changed")
        trace = torch.load(folder/"full_history_trace.pt", map_location="cpu", weights_only=True)
        _, target, _ = targets(data, indices[case["source"]], case)
        if trace["reference_codes_sha256"] != tensor_digest(target):
            raise ValueError("retained target binding changed")
        predictions.append(trace["coordinates"].flatten(1));codes.append(target.flatten(1))
        bindings.append(dict(case=i, source=case["source"], blocks=len(target),
            trace_sha256=sha(folder/"full_history_trace.pt"), header_sha256=sha(folder/"header.json")))
    if sum(x["blocks"] for x in bindings) != 10:raise ValueError("fixed ten blocks required")
    return torch.cat(predictions), torch.cat(codes), dict(source_report_sha256=sha(run/"report.json"),
        source_verification_sha256=sha(run/"verification.json"), bindings=bindings)


def run(out, device):
    predicted, target, bindings = inputs()
    identity = dict(**committed_identity(SOURCES), **bindings, codec_sha256=sha(CODEC),
        device=str(device), torch_version=str(torch.__version__), seed=20268200,
        times=[0., .5, .9], strength=.001, model_updates=0, language_quality_claim=False)
    if out.exists():raise FileExistsError("preserve partial or completed preflight")
    out.mkdir();write(out/"identity.json", identity)
    codec = load_qualified_writing_codec(CODEC, device)
    original = {n:p.detach().cpu().clone() for n,p in codec.state_dict().items()}
    # High precision is an independent numerical reference, not an inference path.
    reference = copy.deepcopy(codec).to(device="cpu", dtype=torch.float64)
    noise = torch.randn(target.shape, generator=torch.Generator(device="cpu").manual_seed(20268200))
    residual = (predicted-target).to(device).requires_grad_(True)
    records = []; retained = {}
    try:
        for t in identity["times"]:
            x = ((1-t)*noise+t*target).to(device).requires_grad_(True)
            synchronize(device);begun = time.perf_counter()
            loss, components = decoder_pullback_error(codec, x, residual, strength=.001)
            gradient, x_gradient = torch.autograd.grad(loss.sum(), (residual,x), allow_unused=True)
            synchronize(device);elapsed = time.perf_counter()-begun
            zero, _ = decoder_pullback_error(codec, x, residual, strength=0.)
            zero_gradient, = torch.autograd.grad(zero.sum(), residual)
            mse = residual.square().mean(-1)
            mse_gradient, = torch.autograd.grad(mse.sum(), residual)
            if not torch.equal(zero, mse) or not torch.equal(zero_gradient, mse_gradient):
                raise AssertionError("zero strength does not reproduce original MSE exactly")
            if x_gradient is not None:raise AssertionError("observed noisy metric point received a gradient")
            if not torch.isfinite(loss).all() or not torch.isfinite(gradient).all():
                raise FloatingPointError("nonfinite physical metric loss or gradient")
            y = x.detach().requires_grad_(True)
            decoded = normalized_decoder_logits(codec,y)
            _, tangent = torch.func.jvp(lambda q:normalized_decoder_logits(codec,q),
                (x.detach(),), (residual.detach(),))
            adjoint, = torch.autograd.grad(decoded,y,grad_outputs=tangent.detach())
            expected = 2*residual.detach()/1536+2*.001*adjoint/2048
            gradient_error = error(gradient,expected)
            if gradient_error["relative_l2"]>2e-4 or gradient_error["maximum_absolute"]>1e-4:
                raise AssertionError("forward JVP loss gradient differs from reverse-mode adjoint")
            with torch.no_grad():
                normalized = F.layer_norm(x.detach().reshape(-1,768),(768,),eps=codec.config.layer_norm_epsilon)
                physical = codec.decode(normalized).reshape(-1,2048)
                if not torch.equal(decoded.detach(),physical):raise AssertionError("physical decoder logits differ")
                xd, vd = x.detach().cpu().double(), residual.detach().cpu().double()
                finite = []
                for step in (1e-3,5e-4):
                    central = (normalized_decoder_logits(reference,xd+step*vd)-normalized_decoder_logits(reference,xd-step*vd))/(2*step)
                    discrepancy = error(tangent.detach().cpu(),central)
                    finite.append(dict(step=step,**discrepancy))
                    if discrepancy["relative_l2"]>2e-4:
                        raise AssertionError("actual FP32 JVP differs from FP64 central differences")
            row = dict(time=t,loss=loss.detach().cpu().tolist(),code_mse=components["code"].detach().cpu().tolist(),
                decoder_tangent_mse=components["decoder_tangent"].detach().cpu().tolist(),
                gradient_norm=float(gradient.norm()),gradient_adjoint_check=gradient_error,finite_difference_checks=finite,
                original_mse_exact=True,noisy_input_gradient_is_none=True,physical_logits_equal=True,
                forward_and_backward_seconds=elapsed)
            records.append(row)
            retained[str(t)] = dict(noisy=x.detach().cpu(),residual=residual.detach().cpu(),tangent=tangent.detach().cpu(),
                loss=loss.detach().cpu(),gradient=gradient.detach().cpu(),adjoint_gradient=expected.detach().cpu())
        if any(p.grad is not None or p.requires_grad for p in codec.parameters()):
            raise AssertionError("codec parameters are not frozen")
        if any(not torch.equal(original[n],p.detach().cpu()) for n,p in codec.state_dict().items()):
            raise AssertionError("preflight changed the fixed codec")
        torch.save(retained,out/"derivatives.pt")
        write(out/"report.json",dict(identity=identity,rows=records,passed=True,
            codec_parameters=sum(p.numel() for p in codec.parameters()),codec_frozen_and_unchanged=True,
            retained_derivatives_sha256=sha(out/"derivatives.pt"),process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            peak_allocated_vram_gib=torch.cuda.max_memory_allocated()/1024**3 if device.type=="cuda" else None,
            timing_scope="first actual decoder-only JVP loss and residual-gradient invocation per time; no full actor",
            no_optimization_or_new_generation=True,language_quality_claim=False,general_ilm_goal_complete=False))
        print(json.dumps(dict(passed=True,rows=records),indent=2))
    except BaseException as exc:
        write(out/"failure.json",dict(error=repr(exc),identity=identity,completed_rows=records,language_quality_claim=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device",choices=("cpu","cuda:0"),default="cpu")
    parser.add_argument("--out",type=Path,required=True)
    args = parser.parse_args();out=args.out.resolve()
    if args.device == "cpu":
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():raise ValueError("explicit CPU-only preflight required")
        torch.set_num_threads(2);run(out,torch.device("cpu"))
    else:
        cpu = ROOT/"artifacts/decoder_pullback_v82_cpu_20261107/report.json"
        if not cpu.exists() or not json.loads(cpu.read_text())["passed"]:raise ValueError("CPU derivative preflight must pass first")
        with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            run(out,setup())


if __name__ == "__main__":main()
