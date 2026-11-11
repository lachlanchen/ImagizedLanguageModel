# Actual glyph-actor native reading diagnostic

Freeze this helper-only comparison before model execution. It implements the
[bounded decision](glyph_tail_native_reader_decision.md); no optimizer update,
new ILM image or deployed source decoder is introduced.

## Fixed comparison

Use all 32 current QA prompts, both full input presentations, and normal,
question-erased and passage-erased images. Each is tested with the learned
tail and with all source contextual adapter outputs zeroed: 384 image calls.
The same full prompts as text add 64 calls across the two tail conditions.
Use one body serially, adapted first, without duplicating shared weights.

CPU preparation authenticates the original 192 image inputs and corresponding
actual glyph-generation archives. Extract the first FP32 contextual vector
from each original archive, with its geometry and source hashes. For every
adapted image call, require bitwise equality to that saved vector before
interpreting the helper answer. Unadapted calls are a separate control, not
purported reproductions of adapted states. All complete text prompts come
from the same authenticated corpus records; no reference answer is passed
to generation.

The helper uses the original source vocabulary projection and source
previous-symbol embeddings as local diagnostic variables. Neither is
registered on the image actor. Keep the unchanged numeric task frame, native
visual observer, rotary convention and deterministic context numerics.
Use 128 source units including a native end decision; preserve capped text,
source IDs, selected logits and all states. This budget differs from 128 ILM
glyphs and does not support a latency-parity claim.

Dequantize the source head to BF16, then project ascending chunks of 4,096
rows with FP32 contractions. Select the largest finite logit, breaking exact
ties by the smaller source ID. Record this procedure rather than claiming
bitwise equivalence to a differently shaped full-vocabulary contraction.
Guard temporary allocations as well as contextual calls. Keep the shared
archive in place and never attach its lexical modules to the ILM.

## Measurements and interpretation

After complete generation, independently verify saved case/source identities,
first-state matches, output text/IDs/stopping and all denominators on CPU.
Use the unchanged whitespace-normalized complete canonical exact match and
learned-stop requirement, 32 questions and sixteen passage pairs per arm.
Both-correct pairs must have distinct complete outputs. Report normal versus
both erased conditions, adapted versus unadapted and image versus text, with
paired passage-group bootstrap intervals (10,000 draws; seed 202611154).
Retain all raw noncanonical answers and capped streams. No new semantic score,
answer shortening, prefix removal or favorable subset is allowed after calls.

Use casewise native-versus-glyph failures to select the next intervention.
Native answers without glyph answers implicate writing/feedback; text-only
success suggests an input gap; native success lost under adaptation suggests
an adaptation tradeoff. Treat these as local diagnostic contrasts with their
uncertainty, not proof of a unique causal mechanism or useful ILM capability.
No helper output, state or development answer becomes training data.

## Budget and preservation

GPU 0 under `artifacts/ilm_gpu0.lock`; one body and one heavy project job.
Keep 48 GiB available host RAM and 20 GiB free GPU for admission, 24 GiB
runtime host reserve and 22 GiB peak allocated GPU ceiling. Each attempt has
3,600 wall seconds including at most 600 pressure-wait seconds. Preserve
partial case directories and attempt receipts. Resume only the identical
registered jobs/sources; include all attempts and setup/wait costs. Do not
interpret partial coverage as the completed comparison. Leave foreign GPU
services and GUI stacks untouched.

The original glyph outputs, compact failed fits and current manuscript remain
unchanged. This diagnosis tests inherited capability through a symbolic
helper; it does not train or qualify an independent visual language model.
