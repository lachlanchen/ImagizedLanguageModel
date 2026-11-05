#!/usr/bin/env python3
"""Offline source-engine sanity check; vocabulary use is confined to this helper."""

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from ilm.visual_lm.blind_vision_transcription import owned_reader, MODEL, check_host_memory, gpu_memory
from ilm.visual_lm.gguf_language_trunk import load_gguf_reader_source
from ilm.visual_lm.gguf_visual_language import QwenVisualLanguageArchive
from ilm.visual_lm.grounded_image_weights import load_grounded_image_actor, file_sha, MODEL_SHA


ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = (
    "Reply with just the number: 2 + 3 = ?",
    "At what temperature in degrees Celsius does pure water freeze at standard pressure? Reply with just the number.",
    "What is the opposite of hot? Reply with one word.",
    'Translate "cat" into Chinese. Reply with only the translation.',
    "只回答数值：七乘以八等于多少？",
    "太阳通常从哪个方向升起？只回答方向。",
    "把“水”翻译成英语，只回答单词。",
    "“木”字通常表示树木还是水？只回答其中一个。",
)
OPTIONS = dict(temperature=0, top_k=1, top_p=1, repeat_penalty=1,
               seed=20266411, num_ctx=2048, num_predict=8)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True)+"\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT/"artifacts/grounded_context_transfer_20261105/check")
    parser.add_argument("--local-root", type=Path, default=ROOT.parent/"LocalLLM")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise ValueError("physical GPU 0 required")
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out/"report.json").exists():
        raise FileExistsError("completed check is immutable")
    lock = (args.out/"operation.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    begun = time.perf_counter()
    sources = [Path(__file__), ROOT/"references/inherited_context_check_plan.md",
        ROOT/"ilm/visual_lm/grounded_image_actor.py", ROOT/"ilm/visual_lm/grounded_image_weights.py"]
    identity = {str(p.relative_to(ROOT)): file_sha(p) for p in sources}

    def stage(name, **values):
        row = dict(stage=name, pid=os.getpid(), physical_gpu=0, elapsed_seconds=time.perf_counter()-begun, **values)
        write_json(args.out/"status.json", row)
        print(json.dumps(row), flush=True)

    try:
        check_host_memory()
        if gpu_memory() > 256:
            raise RuntimeError("GPU 0 occupied")
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        root = args.local_root/".local/models/ollama"
        blob = root/"blobs"/("sha256-"+MODEL_SHA)
        if file_sha(blob) != MODEL_SHA:
            raise ValueError("source archive differs")
        archive = QwenVisualLanguageArchive(blob, load_gguf_reader_source(args.local_root/".local/tools/llama.cpp/gguf-py"))
        from transformers.integrations.ggml import convert_gguf_tokenizer
        from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
        tokens = archive.field("tokenizer.ggml.tokens")
        tokenizer, _ = convert_gguf_tokenizer("qwen2", dict(tokens=tokens,
            merges=archive.field("tokenizer.ggml.merges"), eos_token_id=archive.field("tokenizer.ggml.eos_token_id")))
        decoder = {value: key for key, value in bytes_to_unicode().items()}
        by_bytes = {}
        for index, token in enumerate(tokens):
            if all(c in decoder for c in token):
                by_bytes.setdefault(bytes(decoder[c] for c in token), []).append(index)

        def source_id(row):
            values = by_bytes.get(bytes(row["bytes"]), [])
            if len(values) != 1:
                raise ValueError(f"source token bytes ambiguous or missing: {row}")
            return values[0]

        prompts = ["<|im_start|>user\n"+q+"<|im_end|>\n<|im_start|>assistant\n" for q in QUESTIONS]
        prompt_ids = [tokenizer.encode(p, add_special_tokens=False).ids for p in prompts]
        if any(tokenizer.decode(ids, skip_special_tokens=False) != prompt for ids, prompt in zip(prompt_ids, prompts)):
            raise ValueError("source tokenizer prompt round trip differs")
        stage("reference_engine")
        references = []
        actual_reference_calls = 0
        with owned_reader(ROOT/"artifacts/question_assignment_v63_20261105", args.out/"reference_runtime", args.local_root, 11437) as reader:
            for index, prompt in enumerate(prompts):
                payload = dict(model=MODEL, prompt=prompt, raw=True, think=False, stream=False,
                    options=OPTIONS, logprobs=True, top_logprobs=10, keep_alive="1m")
                path = args.out/"reference"/f"{index}.json"
                if path.exists():
                    stored = json.loads(path.read_text())
                    if stored["request"] != payload:
                        raise ValueError("partial reference request differs")
                    response = stored["response"]
                else:
                    result = reader.session.post(reader.base_url+"/api/generate", json=payload, timeout=(5, 180))
                    result.raise_for_status()
                    response = result.json()
                    actual_reference_calls += 1
                    write_json(path, dict(request=payload, response=response))
                if response.get("prompt_eval_count") != len(prompt_ids[index]):
                    raise ValueError("source/native prompt counts differ")
                positions = response.get("logprobs", [])
                if not positions or any(not p.get("top_logprobs") for p in positions):
                    raise ValueError("missing reference positions/probabilities")
                references.append(dict(prompt_ids=prompt_ids[index], next_ids=[source_id(p) for p in positions],
                    positions=[dict(ids=[source_id(t) for t in p["top_logprobs"]],
                        probabilities=[math.exp(t["logprob"]) for t in p["top_logprobs"]]) for p in positions]))
                stage("reference_engine", completed=index+1)
        cleanup = json.loads((args.out/"reference_runtime/cleanup.json").read_text())
        if cleanup["remaining_owned_live_pids"] or cleanup["port_open_after_cleanup"] or gpu_memory() > 256:
            raise RuntimeError("owned reference engine did not release resources")

        stage("native_context_loading")
        actor, provenance = load_grounded_image_actor(ollama_root=root,
            gguf_source=args.local_root/".local/tools/llama.cpp/gguf-py", device=torch.device("cuda:0"),
            progress=lambda count, total: stage("native_context_loading", loaded=count, total=total))
        actor.eval()
        if any(bool(p.any()) for name, p in actor.named_parameters() if name.endswith(".up")):
            raise ValueError("context adapters must start at zero output")
        # Temporary helper head only. Never attach it or source token rows to
        # the actor; no deployment/training checkpoint is written by this helper.
        head_tensor = archive.omitted["output.weight"]
        head = torch.from_numpy(np.array(archive.gguf.dequantize(head_tensor.data, head_tensor.tensor_type), copy=True)).to("cuda:0", torch.bfloat16)
        embedding = archive.omitted["token_embd.weight"]
        rows = []
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for index, reference in enumerate(references):
                ids = reference["prompt_ids"]+reference["next_ids"][:-1]
                values = archive.gguf.dequantize(np.array(embedding.data[ids], copy=True), embedding.tensor_type)
                x = torch.from_numpy(np.array(values, dtype=np.float32, copy=True)).to("cuda:0")[None]
                output = actor.context(inputs_embeds=x, use_cache=False).last_hidden_state[0]
                states = output[len(reference["prompt_ids"])-1:]
                logits = F.linear(states.float(), head.float()).float()
                logprob = logits.log_softmax(-1)
                predicted = logits.argmax(-1).cpu().tolist()
                positions = []
                for j, ref in enumerate(reference["positions"]):
                    p = np.array(ref["probabilities"], dtype=float)
                    if not np.isfinite(p).all() or p.sum() > 1+1e-5:
                        raise ValueError("reference probabilities invalid")
                    q = logprob[j, ref["ids"]].exp().cpu().numpy().astype(float)
                    # Correct only a possible rounding excess above unit mass;
                    # otherwise retain all unreported probability in its bin.
                    p /= max(1., p.sum())
                    q /= max(1., q.sum())
                    p = np.append(p, max(0., 1-p.sum())); q = np.append(q, max(0., 1-q.sum()))
                    positive = p > 0
                    kl = float(np.sum(p[positive]*np.log(p[positive]/np.maximum(q[positive], 1e-300))))
                    positions.append(dict(reference_id=reference["next_ids"][j], native_id=predicted[j],
                        agreed=predicted[j] == reference["next_ids"][j], coarsened_kl_nats=kl))
                rows.append(dict(index=index, question=QUESTIONS[index], prompt_tokens=len(reference["prompt_ids"]),
                    positions=positions, logits_finite=bool(torch.isfinite(logits).all())))
                stage("native_context", completed=index+1)
        all_positions = [p for r in rows for p in r["positions"]]
        first = float(np.mean([r["positions"][0]["agreed"] for r in rows]))
        agreement = float(np.mean([p["agreed"] for p in all_positions]))
        kl = float(np.mean([p["coarsened_kl_nats"] for p in all_positions]))
        checks = dict(eight_cases=len(rows) == 8, first_token_agreement=first >= .75,
            reference_position_agreement=agreement >= .85, coarsened_kl=kl <= .05,
            finite_logits=all(r["logits_finite"] for r in rows), actor_vocabulary_absent=actor.context.embed_tokens is None,
            source_engine_released=True, source_identity_unchanged=identity == {str(p.relative_to(ROOT)): file_sha(p) for p in sources})
        write_json(args.out/"native_positions.json", rows)
        report = dict(checks=checks, passed=all(checks.values()), first_token_agreement=first,
            reference_position_agreement=agreement, coarsened_kl_mean_nats=kl, positions=len(all_positions),
            source_sha256=identity, provenance=provenance, peak_allocated_vram_gib=torch.cuda.max_memory_allocated()/1024**3,
            elapsed_seconds=time.perf_counter()-begun, source_response_cases=8,
            actual_reference_calls=actual_reference_calls, actor_image_generation_calls=0,
            helper_only_vocabulary=True, language_quality_claim=False, general_ilm_goal_complete=False,
            scope="Eight text-only source-engine sanity cases, not full equivalence or image-language quality.")
        write_json(args.out/"report.json", report)
        stage("complete", passed=report["passed"])
    except BaseException as error:
        write_json(args.out/"failure.json", dict(error=repr(error), source_sha256=identity))
        stage("failed", error=repr(error))
        raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    main()
