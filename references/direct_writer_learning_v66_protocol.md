# Direct image writing over the complete Chinese/English corpus

Registered 5 November 2026, after V65 and before this run. V65's matched
fitting experiment shows that the actual writer can recover 63/64 images
from inherited features using direct regression, while its flow route fails
even with complete target-pixel conditions. This motivates a direct output
bootstrap. Finite fitting is not a generalization or language result.

## Candidate and inheritance

Keep the complete authenticated grounded actor resident: all 27 vision and
36 contextual layers, adapters, queries and stopping head. Freeze all except
the same 19,374,096-parameter writer. Its new `DirectBlockWriter` uses the
unchanged writer architecture with zero physical-image/time inputs and one
forward pass. It predicts the next physical image without a character
inventory, OCR, font renderer or model-service call. This stage receives
source-answer-image features for reconstruction, not unknown future pixels
for autonomous language generation.

Initialize only the writer from V65 inherited/direct final EMA, SHA-256
`f9c5d846a280a9340237c0a52b1fb7f893cd533e4a1ce8bc3c00f1262fc1a3a1`.
Its previous 2,000 updates on 64 training images are part of provenance.
Require the completed independent V65 verification receipt before loading.
The already verified full-body preflight/context checks remain prerequisites.
Do not modify the frozen V64/V65 implementations or their results.

## Actual learning

Use all 12,976 examples in the complete-writing training cache, report SHA
`0ffcfd101ea26d79c4084bfd166bd146ff5560f5fa83b7aa57c6b48558377d7e`.
For each of 10,000 updates, draw eight CMRC and eight Dolly examples uniformly
within source, then one complete 32-by-64 answer block per example uniformly.
Exclude only the two structural termination blocks; actual white space remains.
The frozen FP32 encoder reads every sampled image independently and returns
two 4,096-vectors. Repeated cases remain valid samples. No truncation to a
finite fitting subset or corpus-specific character vocabulary is allowed.

For observed source-image condition c and signed target Y, minimize

```text
mu_theta(c) = writer(zeros(32,64), time=0, c)
L_direct = mean_pixels ||mu_theta(c) - Y||^2.
```

There is no injected generation noise or integration schedule. The regression
estimates a conditional mean; it does not by itself represent multiple valid
language continuations. A functioning prompt-dependent language actor and
appropriate conditional-choice tests remain necessary after this bootstrap.

AdamW 1e-4, betas (.9,.95), weight decay .01, clip norm 1.0; linear warmup
500 updates, cosine decay to 1e-5; FP32 trainable weights/moments, BF16 writer
contractions, deterministic math attention, EMA .99. Seeds: model/global
20266620, CPU image sampling 20266621. No noise-generator seed is needed.
Save current/previous checkpoints, optimizer, writer/EMA and all sampling/global
RNG state every 250 updates, at update two and on handled interruption. Replay
update three from update two exactly before proceeding, counting the extra
optimizer replay. Retain full sample/metric journals and final fixed EMA.

## Fixed development measurement

After final EMA fixation only, use the exact V64 selection procedure and seeds
for 128 reused-development blocks (64 CMRC, 64 Dolly), with development cache
report SHA `14b088cff0c91175570f4635af7363419225cb36c8784d11efa679b0993bb369`.
Check that the new manifest matches V64's saved manifest byte for byte.
This allows a descriptive comparison; the different objective and warm start
preclude treating it as an otherwise matched broad-corpus intervention.

Generate each image once with its correct visual condition and once with the
same one-case cyclic permutation used by V64. A direct writer uses one pass
and a fixed zero threshold. Report every raw foreground F1, exact-block/blank
fraction, per-case arrays and first-eight images without repair or selection.
Require the unchanged component gates: F1 >=.90 and mean correct-minus-shuffled
gain >=.20. Both must pass. Keep exact reconstruction distinct from reading,
semantic understanding, prompt following and historical-form generalization.
No sealed data is opened and no development output is inspected before the
final EMA. The development partition has been reused and is not a fresh test.

If the component qualifies, register complete prompt-dependent instruction
learning, including stopping and autonomous question controls, before its
quality audit. If it fails, preserve the result and diagnose the failure.
Neither outcome completes or narrows the independent ILM goal.

## Execution and resources

Entry point: `scripts/train_direct_grounded_writer_v66.py`. Commit and push
protocol, direct writer module and trainer before starting the owned
`ilm-direct-writer` tmux job. Only physical GPU 0 is allowed; GPU 1 services
remain untouched. Require idle GPU 0 at startup, host available RAM >=24 GiB,
swap use <75%, allocated CUDA memory <22 GiB and elapsed stage time <24 hours.
Persist safely on signals/resource interruption. Preserve failures. Record
maximum allocated memory across process resumes; retain the full contextual
body even though it is unused in this reconstruction objective. Local compute
includes warm-start and diagnostic provenance; external foundation pretraining
is attributed separately. No frontier-parity or efficiency claim follows.
