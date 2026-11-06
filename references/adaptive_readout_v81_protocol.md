# A matched repair of continuous image-language readout

Prospective protocol, 6 November 2026, after the completed native V80 audit and
fixed 32-question diagnosis. That model fails even with correct image history:
50.241% foreground F1 and no exact images, versus 99.9953% F1 from supplied
codes. This experiment tests conditional representation adaptation; it does
not assume that adaptation will solve the failure.

Reuse all 32 authenticated training questions in the existing sixteen
same-passage pairs, in their original order. No case selection, new labels,
development tuning or answer changes are permitted. Both arms warm-start the
same completed V80 image input, writing and stop weights, checkpoint SHA-256
`2e267b023118552e2462971c5f8e0dc49b57f4e401ef7ba58082c05fc85fc0f5`.

The control continues the unchanged frozen native transition. The adapted arm
adds one learned continuous initial query and trains the already allocated
rank-8, alpha-16 q/v/o adapters in all 36 contextual layers. This is a composite
intervention. Both retain the same image codec, full prompt KV, clean-code
objective, uniform time law, 32-step sampler, actual-pixel feedback and learned
stop. No language vocabulary, teacher service, target code lookup or supplied
length enters autonomous generation. Complete parameter count includes the
externally pretrained body; this is not training a lightweight model from
scratch.

For fixed prompt cache K0(X), the adapted condition is the last causal state
of F_(base,adapter)(K0(X),q,A(E(Y1)),...,A(E(Yj))). The query consumes one actual
cache/rotary position; each subsequent block consumes two. The original
prompt cache is computed with adapter effects disabled. Adapter scope must be
entered inside each suffix layer invocation and its checkpoint recomputation,
so changing adapters cannot stale K0(X). The true prefix remains immutable.

Before optimization, a real-model preflight must verify prompt-memory identity
against the authenticated training cache, adapter-invariant prompt prefill,
finite nonzero query/input/writing/stop/adapter gradients, full-history versus
incremental conditions, future exclusion, right-padding and exact optimizer,
sampler and noise-state resume. Hidden-state numerical comparisons use a
prospectively fixed maximum absolute tolerance of 1e-4; source memory identity
and resumed model updates require exact equality. Verify actual emitted-pixel
feedback, frozen-weight and prompt-cache integrity. Record failures and peak
allocated memory. A preflight exceeding 24 GiB does not qualify this route for
one card. Reset preflight optimization before each experimental arm.

Run arms sequentially, with a single shared foundation installation and the
project GPU0 lock. Prefill the 32 real prompt images into CPU RAM once per
invocation. Train each arm for exactly 2,000 updates, batch four, AdamW
(0.9,0.95), weight decay 0.01, clip one, peak learning rate 1e-4, 100-update
linear warm-up and cosine decay to 1e-5. Fresh optimizer states are used for
the repair. Seed 20268100; exact shuffled 32-case sampler seed +1; CUDA noise
generator seed +2; query initialization uses its own CPU generator seed +3.
Both arms receive the same case order and noise draws. Checkpoints include
all trained adapters/query, interfaces, optimizer, sampler and RNG states,
without duplicating frozen weights. Preserve current and previous checkpoint.

Retain autonomous answers for all 32 questions before training and after the
final update. Also retain correct-history/known-length control outputs and
supplied-code reconstructions at the final update. Use the original diagnostic
seeds, 128-block cap and all original targets. No best-checkpoint selection or
early stopping from these answers. Independently score the emitted pixels and
decode the retained control coordinates. The provisional repair criterion is
at least 24/32 pixel-exact free answers and mean free foreground F1 at least
0.98. The adapted arm must also exceed the matched control; report all paired
results, not only this threshold. This is a training-fit requirement, not
evidence of general language ability. Failure prevents scaling this recipe
without a new diagnosis; success still requires new-prompt/copy and original
development evaluation before a usable ILM can be claimed.

Run from the repository root with the compatible existing environment:

```bash
CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=4 \
PYTHONPATH=. artifacts/external/mixar_env_py312/bin/python \
  scripts/train_adaptive_readout_v81.py --stage preflight
```

Only after the actual preflight passes, use the same GPU environment with
`--stage train`. Resume uses the same command and authenticated checkpoint;
retain failures and investigate unbound partial output. Then run `--stage
verify` with `CUDA_VISIBLE_DEVICES=''` and the same compatible Python. The
verification must reproduce all 32 original control generations exactly,
independently score every final answer and replay the supplied/control decodes.
All runtime outputs default to `artifacts/adaptive_readout_v81_20261106`.
