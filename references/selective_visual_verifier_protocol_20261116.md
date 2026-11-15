# Training-only content verification with explicit abstention

The [whole-answer image reward](whole_answer_learning_direction_20261116.md)
fails its fixed calibration because an output with an uncertain frozen reading
outranks a known-correct answer. Earlier recognition guidance also tolerated
duplicated glyphs and whole lines. A training reward needs to account for
every emitted position without supplying question context to its recognizer.
This protocol tests one replacement verifier. It does not train a student.

## Controlled change

Replace foreground-Dice substitution with selective recognition of individual
physical glyph images. Build an offline inventory from **all 108 training
pixel shards** of the existing corpus. Deduplicate complete 32-by-32 binary
bitmaps. Do not use development images, reference answer images, Unicode labels,
teacher alignment IDs, fonts or a language model to construct the inventory.
Do not select only frequent or successful training shapes.

For distinct prototypes c_i, compute

\[
 \delta_i=\min_{j\ne i}d_H(c_i,c_j),\qquad
 i^*(x)=\arg\min_i d_H(x,c_i).
\]

Accept i*(x) only when

\[
 2d_H(x,c_{i^*})<\delta_{i^*};
\]

otherwise abstain. These strict Hamming balls are disjoint by the triangle
inequality. This is a bounded-corruption geometric property, not a theorem
that every accepted arbitrary image is readable or has a known semantic
identity. Enough corruption can move a source image into another prototype's
ball. A font shift or a useful unfamiliar glyph may be rejected.

Recognition receives only generated cells. Reference images cannot influence
their identities. Exact reference images must already exist in the training
inventory; missing forms fail coverage and must not be added from development.
Equal physical forms share an identity, so the verifier cannot resolve distinct
characters that have identical rasters. The component is a finite image
recognizer despite using no Unicode labels. It is explicitly training-only:
never bundle it into the ILM or use it to select, repair or rerender answers.

Score stopped answer streams with the same complete-answer bonus plus
normalized ordered edit similarity used in the previous surrogate, now over
accepted physical identities. Every abstention is a mismatch. Keep real blank
cells, repeated glyphs, insertion/deletion and order. Capped and wholly blank
answers receive zero. The critic does not decide which linguistic answer is
correct; the separate training reference supplies that supervision.

The [implementation](../ilm/training_only/visual_content_verifier.py) uses
exact integer-valued Hamming distances computed with binary float32 dot
products on CPU. Prototype IDs never encode linguistic labels.
Seven [CPU checks](../tests/test_training_only_visual_verifier.py) pass,
including exhaustive 8-bit corruption of a physical fixture, exact/tied
decoding, preserved whitespace/order/duplication, unseen-reference abstention,
and invariance of recognition to which reference is scored. These fixtures
prove implementation contracts only.

## Fixed real-output comparison

The [registration](evidence/selective_visual_verifier_registration_20261116.json)
binds 347 source/input hashes and the same 64 final normal QA outputs used by
the rejected image reward. Keep all 9 strict complete-correct, 9 readable-wrong
and 46 uncredited/uncertain outcomes. Original readings stay frozen; do not
repair them from a new recognition or source reference. A conservative failure
against uncertainty does not establish that the uncertain reading is wrong.

Require all clean references covered, all nine correct complete contents
accepted, no readable-wrong or uncertain complete content accepted, known
correct/readable-wrong AUC at least .95, every correct reward strictly above
every uncredited reward, and all capped rewards zero. These prospective checks
do not turn this reused model-reviewed cohort into an independent validation.
There is no independent human adjudication. Preserve all disagreements.

Commit and push implementation/registration before running the CPU comparison.
Use two BLAS threads, at most 2 GiB RSS and 300 seconds of calibration time,
retaining at least 24 GiB available host RAM. Hash authentication and imports
precede the calibration timer and must not be hidden in an efficiency claim.
The single CPU process saves one small derived training inventory; it acquires
no weights, invokes no neural model and uses no GPU. Current white-target
training sources, ownership, schedule and gates are unchanged. An error leaves
its partial artifacts for inspection; no automatic retry or replacement run.

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=2 PYTHONPATH=. \
  python scripts/calibrate_selective_visual_verifier.py register
# Commit and push the registration, then:
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=2 PYTHONPATH=. \
  python scripts/calibrate_selective_visual_verifier.py run
```

Failure rejects this verifier for policy optimization. Do not relax the radius,
add held-out prototypes, change frozen readings or launch its policy fit. A
pass is only a limited content-verification result. It still requires an
adversarial/support/resource preflight on actual sampled student outputs,
after the current serial model run finishes and under separate registration.
No model training is authorized by this protocol. Useful complete language,
independent image-only generation and matched-quality resource savings remain
unproved.
