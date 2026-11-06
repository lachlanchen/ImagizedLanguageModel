# Retain contextual suffix information without direct visual-key access

V77's source helper answers 8/12 copy requests and 19/32 fixed training
questions exactly, while the image writer gets zero in both cohorts. All44
normal-image final-layer suffix memories match, but the helper continues with
full layerwise visual/context KV history. Before replacing the writer, test
whether the 52 suffix positions' layerwise KV states retain answer content.
This diagnostic is not a new image-output language model or a training run.

## Fixed intervention

Reuse all44 V77 normal-image inputs in their original order. Authenticate its
completed preparation, generation and CPU-verification receipts. Reuse its
same observer pixels, numeric task controls, frozen source weights, zero
adapter outputs, BF16 initial projection state, greedy source decoder, end
IDs and64-token cap. Use seed20267710 and the same deterministic physical
GPU0 configuration. There are44 new masked helper responses; the original
44 full-cache responses are frozen comparators and are not regenerated.

After unchanged prefill, retain the full cache tensor lengths and all original
sequential/rotary coordinates. For subsequent one-query continuations, pass
an explicit FP32 four-dimensional additive attention mask: positions before
the52 task-suffix keys receive the minimum finite value, while those52 keys
and every generated continuation position remain visible. The first decision
still uses exactly the original stored BF16 final state and must match V77.
No keys are physically deleted in this experiment. A general mask utility
also excludes future positions in parallel multi-query calls; its causal
geometry must be checked on CPU before GPU use.

Before the44 generations, use the first case's prefill and its actual first
greedy choice for four single-step forward checks. Compare original cached
keys/values against a cloned cache with all excluded keys filled with7 and
values with-11, preserving every suffix value. The two masked next hidden
states must be bit-identical, while the two unmasked states must differ by
more than1e-5. The unmasked original next state must match V77's retained
second state within1e-5. Retain both caches and all four state vectors for
independent CPU checks. These four forced single-step checks receive no
autonomous-answer credit. The chosen input is the prior source decision,
not a target answer. The source cache and weights must remain unchanged.

Measure actual full and52-position layerwise KV bytes, shapes and dtypes
before decoding each case. Final-layer BF16 memory and multilayer KV are
different interfaces and costs. Keep the temporary source tokenizer, input
row lookup and output projection in this offline evaluator only; do not
attach them to or export them with the deployed ILM. No optimizer updates,
teacher-produced training data, image-actor calls, sealed access, new model
downloads or output repairs occur. Preserve every response, including failures.

## Assessment and verification

Report the same stopped canonical exactness, edit accuracy, reference
inclusion/selectivity, stopping and truncation as V77, separately for copy
and training cases. Report exact retained-response/ID agreement with the
corresponding full-cache calls, all first-decision agreements, actual KV
bytes and the memory ratio. This descriptive intervention has no new general
language gate. Retention of full-cache-correct answers and loss of those
answers are reported case by case, not only as aggregate counts.

Independent CPU verification authenticates original and new inputs/outputs,
checks all44 prefill memory hashes and source-first decisions, reconstructs
the mask and every visible/blocked region, validates that cache poisoning
changes only excluded positions, and compares retained preflight vectors.
It decodes all source IDs, recomputes every score and replays all temporary
source projections with the unchanged V77 tolerances: selected-logit error
at most1e-3 and source-choice regret at most1e-4; exact argmax agreements and
near ties are separate. It does not independently replay the complete neural
body. Pin the installed attention/masking/cache source-file hashes and versions
in addition to committed project code. Commit and push before execution.

If content survives this visibility restriction, it supports testing a compact
multilayer KV interface with learned image-history input and continuous image
output adapters. It does not prove physical cache-deletion equivalence,
final-layer-only memory sufficiency, unencoded-form generation, a complete
independent ILM or efficiency relative to text LLMs. If content is lost, retain
or retrieve more visual context before another writer-training run. Use one
project-owned GPU0 job and the shared lock; release it after measurement.
