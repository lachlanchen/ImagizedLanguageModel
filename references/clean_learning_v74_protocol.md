# Paired image-language learning control

This control tests whether the complete clean-code writer can learn different
Chinese answers to different questions about the same visible passage. It uses
seen training examples to diagnose learning capacity and autonomous writing.
Passing would not establish unseen-question generalization, historical-script
competence or a usable general ILM. The original full development audit and
sealed examples are not used to select cases, checkpoints or settings.

## Fixed examples and inputs

Use the authenticated 12,976-example image-memory cache from the V72 run and
the admitted CMRC training source. Group eligible training records by the
SHA-256 of the literal visible context, sort the groups by that digest, then
sort candidates by source name. Within each group choose the first pair with
different questions, different whitespace-stripped reference answer text,
and different physical target digests. Select the first 16 eligible groups:
32 questions in total. The immutable plan records the exact inventory and
source/cache/pixel digests before optimization. The supplied-code reference
images must also differ within every selected pair.

Text is used only for offline selection. Optimization receives frozen
prompt-image features and continuous codes derived from answer images.
The actor receives only the owning folded prompt pixels. No answer, source
name, question identity, text, target length, glyph inventory or teacher is
an inference input. Existing foundation weights are explicitly attributed;
this is not a claim of learning language from scratch on 32 examples.

## Fixed training

Initialize the complete 64,520,193-parameter CleanSemanticWritingModel afresh.
Do not initialize from the one-image control. Train for 4,000 updates,
batch 16, using exact shuffled passes over all 32 selected examples:
64,000 draws and 2,000 complete selected-set passes. Use AdamW, learning
rate 0.0001, betas (0.9, 0.95), weight decay 0.01 and gradient clipping 1.
Warm up for 200 updates, then cosine decay to one tenth of the peak rate
over the remaining 3,800 updates. Model, sampler and noise seeds are
20267410, 20267411 and 20267412.

The causal conditions use the prompt memory and previous answer-image codes.
For each example sample at most 16 answer-block positions, weighted equally
per example. For clean target y and Gaussian noise epsilon, train on
z=(1-t)epsilon+t*y with t=sigmoid(-0.8+0.8*u), u standard Gaussian.
Minimize the mean squared clean-code error divided by max(1-t,0.05)^2,
plus the balanced continuing/terminal stopping loss. The flow body uses BF16;
the final writing projection executes in FP32. Clean-data prediction and the
noise weighting/time distribution follow prior denoising work, including
[Back to Basics](https://arxiv.org/abs/2511.13720), rather than constituting
a new learning-theory result here.

Before production, check exclusion of future target blocks, bit-exact
optimizer/RNG/data-cursor restoration and a forced two-block physical-feedback
branch. These are execution checks, not quality results. Reset the model,
optimizer and all random streams after this preflight. Preserve checkpoints
every 500 production updates; evaluate only the final 4,000-update state.

## Autonomous answer generation

Load the completed writer, frozen visual reader and frozen writing codec
in a separate process on physical GPU 0. Generate all 32 full answers from
actual source prompt pixels, not cached prompt features. Use the same seed
within each pair, 20267420 plus its pair index, 32 analytic endpoint updates
per block, a 128-block safety cap and the learned stopping probability at
0.5. Do not supply target answer lengths or stop positions. Re-encode the
actual emitted binary pixels as causal history after each block.

Record raw answers, source input pixels, reader memory, stopping probabilities
and feedback hashes before scoring. Compare the actual reader memory with
the training cache as a numerical diagnostic; do not substitute cached
features at inference. Gold target access and rendering occur only after the
corresponding actor call. Score the owning original target and the other
question's target from the same pair. This is 32 actual generations; cross
scoring is not an additional model call.

The final control passes only if all fixed requirements hold:

| Requirement | Threshold |
| --- | ---: |
| Mean foreground F1 against original images | at least 0.99 |
| Worst foreground F1 against original images | at least 0.98 |
| Exact full answers against supplied-code reference images | at least 0.90 |
| Pairs with both supplied-code answers exact | at least 0.875 |
| Learned stopping fraction | at least 0.95 |
| Mean owning-target F1 minus cross-target F1 | at least 0.10 |

Also report original-image exact matches separately. The supplied-code
reference is a frozen codec reconstruction, not a substituted original
target; the original-image F1 requirements still apply. Independent CPU
verification reconstructs source input layout, authenticates raw outputs,
re-renders the frozen reference decoder and recomputes all physical scores.
It does not claim to replay the entire neural sampler or every feedback
encoding. A successful file-integrity check cannot override a failed quality
requirement. Preserve all failures and do not tune on this audit.

Run with the compatible project Python and PYTHONPATH=.:

```bash
artifacts/external/mixar_env_py312/bin/python scripts/run_clean_learning_v74.py \
  --run artifacts/clean_learning_v74_20261106
```

The coordinator serializes prepare, training, actual-image audit and CPU
verification. It uses the shared project GPU lock and creates no GUI stack.
For standalone PNG inference use scripts/infer_clean_ilm_v74.py with the
completed checkpoint. Its output remains an experimental model answer.

After a pass, training on the full admitted corpus and passing the original
unseen full-answer audit are still required before a language capability
claim. The general ILM objective remains open after this control.
