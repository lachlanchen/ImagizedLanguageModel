# Qualification of an isolated-cell training reader

The complete white-target trial improved the appearance of individual forms
without qualifying complete language. Earlier whole-answer rewards also failed:
geometric similarity did not conservatively separate uncertain writing from
correct answers, and a selective physical-prototype verifier rejected two known
correct answers while accepting an uncertain one. A previous page recognizer
sometimes omitted repeated writing. These results motivate one bounded change:
transcribe each emitted physical cell independently with a fixed vision teacher,
then compare the ordered readings with the complete training answer.

This is a qualification experiment for a possible **training-only content
signal**. It trains no student and changes no deployed inference path. It is
neither an ILM language result nor an independently adjudicated semantic judge.

## Fixed reader and physical input

Use the installed `qwen3.5:9b-q8_0`, manifest
`a8ee95bc3592ebaa624014ca8b429d25e0ef5f45030dc35786da7ff7e9e1b60a`,
through local Ollama 0.32.6. Bind its template, parameters, model information and
capabilities at registration. This teacher has 8,953,803,264 parameters according
to the installed metadata. Its costs remain separate from student training and
from any claim of independent or lightweight deployment. Shared model family
and unknown inherited exposure prevent a claim of independent review.
The [Qwen model card](https://huggingface.co/Qwen/Qwen3.5-9B) describes vision
support; its published benchmarks do not qualify this local quantization or task.

The request contains only a fixed instruction and one original binary 32×32
cell, enlarged eightfold by nearest-neighbor sampling, with a 16-pixel white
margin. There is no font replacement, stroke repair, question, reference answer,
source identifier, neighboring glyph or conversation history. Ambiguous forms
must produce an abstention. Exactly white physical cells become spaces without
a model call. Every other response must identify one visible Unicode character;
malformed, truncated, foreign, invisible, multiple-character or uncertain replies
remain abstentions. This narrow calibration does not support unencoded writing.

Request a JSON schema, temperature zero, seed 20261116, thinking disabled,
2,048 context positions, at most 96 generated tokens and a 60-second model
keep-alive. These are documented [native Ollama chat
options](https://docs.ollama.com/api/chat); structured syntax does not guarantee
correct reading. Use no hosted service, new daemon, weight download or change to
the shared LocalLLM configuration.

## Cohort and information separation

Retain all 64 final normal QA outputs in the original reward registration,
32 fixed-cell and 32 proportional. Authenticate their saved raw pixels, original
images, reports, review freeze and source reference shards against the earlier
registration. The 9 strict correct, 9 readable wrong and 46 uncertain/uncredited
labels remain unchanged. These outputs are reused development evidence, excluded
from fitting or prompt selection. The untouched DRCD confirmation pool is absent.

All 47 naturally stopped trajectories contribute their 297 actual emitted
positions. The 17 capped trajectories retain a zero reward without transcription
of cap-only cells. Include every clean reference cell as a coverage control.
Deduplicate only exact physical masks and submit all nonwhite masks in global
hash order. The teacher is not told whether a mask came from a correct output,
an error or a clean reference. Cache the first response once; no retries,
alternative prompts, threshold search or revision of frozen old readings.

The CPU registration writes a private pixel inventory and fixes its hash and
exact request count. Commit and push the executor, parser, checks, protocol and
registration before any chat inference. Freeze all raw first responses before
the separate scoring action reads reference strings and old review labels.
Metadata API requests at registration perform no chat inference. Unchanged
source and teacher identities are checked again after transcription.

## Ordered score and prospective rejection

For naturally stopped writing, concatenate the accepted cell readings in their
original order. If any nonwhite cell abstains, or the answer is empty/all white,
assign zero. Otherwise compute ordered character edit distance against the
complete answer, preserving insertion, deletion, repetition and actual spaces.
Physically white reference cells are represented as spaces. Let d be this
distance and m the larger complete extent. The proposed scalar signal is

\[
 r=0.75\max(0,1-d/\max(m,1))+
   0.25\mathbf 1[\operatorname{stripspace}(\hat a)
                   =\operatorname{stripspace}(a)].
\]

The exact-content bonus uses the original scorer's whitespace-only
normalization. It does not normalize punctuation, digits, order or duplicates.
Extra whitespace still affects the edit term. All capped trajectories score
zero, without inventing a natural stop. This is a canonical-answer signal for
fixed-answer QA; valid paraphrases can lose credit. It is unsuitable as a unique
target reward for open-ended book continuation.

Require every clean reference to be transcribed correctly; all nine known
correct answers to receive complete-content acceptance; no readable wrong or
uncertain output to receive complete-content acceptance; correct-versus-readable-
wrong pairwise AUC at least .95; minimum correct reward strictly above every
uncredited reward; and every capped reward zero. All conditions are fixed before
inference. Uncertain old readings are not established semantic errors: rejecting
a judge for disagreement preserves that uncertainty rather than repairing it.

Any failed condition rejects this fixed judge for optimization. Do not tune it
on this cohort or launch a policy fit. Even a pass qualifies only this reused
calibration; a separately registered training-only actor sampling, probability-
replay, gradient and resource preflight is still required on a fresh clone.
Equal rewards or uniformly unreadable actor samples fail that learning-signal
preflight. No student optimizer update is authorized here.

## Resources and preservation

One serial client owns `artifacts/ilm_gpu0.lock`. Require 48 GiB available host
RAM, 20 GiB free GPU0 and 48 GiB free disk at entry; retain 24 GiB host RAM and
48 GiB disk. Use two client CPU threads, at most 2 GiB client RSS and 1,800 seconds
for the transcription action. The cold request timeout is 180 seconds; later
requests are limited to 60 seconds, each also bounded by remaining total time.
Preserve any partial response and stop on a transport/resource/identity error;
there is no automatic recovery or second worker.

Sample physical GPU use before and after requests, with a 22 GiB GPU0 occupancy
ceiling and an abort if another device grows by more than 512 MiB. This sampling
does not measure allocator peak or prove exclusively GPU0 placement. Disclose
actual placement from service/process observations. Refuse a resident shared
teacher model at admission; do not unload another project's model. The short
cache lifetime releases this model naturally after requests. Verify expiry and
leave the shared daemon running. Preserve raw local requests' hashes, responses,
first freeze, resource observations, partial failures and original ILM images;
keep runtime identities and private logs outside Git.

```bash
PYTHONPATH=. OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -m unittest tests.test_isolated_glyph_judge -v
PYTHONPATH=. OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python scripts/calibrate_isolated_glyph_judge.py register
# Commit and push the fixed registration before the next action.
PYTHONPATH=. OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python scripts/calibrate_isolated_glyph_judge.py transcribe
PYTHONPATH=. OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python scripts/calibrate_isolated_glyph_judge.py score
```
