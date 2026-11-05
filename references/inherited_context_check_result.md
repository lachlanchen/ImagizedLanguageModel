# Inherited contextual weights: bounded source-engine sanity result

5 November 2026. The offline check passes its registered requirements. On
eight fixed short English/Chinese cases, the imported body matches **8/8
first-token choices** and **9/9 returned reference positions**. Mean
KL(reference || native) on the reference top-ten bins plus an outside-mass
bin is **0.00026376 nat**. All prompt token counts and tokenizer round trips
match; logits are finite. The source replies contain only one or two returned
positions per case, so this is a small initialization check.

Both engines use the same authenticated quantized archive and raw prompts.
The native helper reads source token rows and a temporary output head outside
the actor, with contextual adapters at zero output. It never attaches that
head to the actor or writes a training/deployment checkpoint. The actor's
vocabulary module stays absent. No generated image or language-quality score
is produced. No development or sealed data is opened.

The owned reference service completes eight requests and exits before native
model loading. Its runner also exits and port 11437 closes; the shared LocalLLM
service is untouched. The whole check takes 103.832 s and peaks at 18.701 GiB
allocated memory including its temporary helper head. This is separate from
the image actor's measured 17.181-GiB training preflight.

The result supports using this contextual initialization. It does not prove
full engine equivalence, visual prompt understanding, open-ended language,
or readable image generation. The next actual learning stage trains the
continuous writer, followed by the required instruction-language experiment.

See the [registered plan](inherited_context_check_plan.md),
[compact receipt](evidence/inherited_context_check.json) and
`scripts/check_inherited_context.py`. Full raw requests/responses and native
position comparisons remain under
`artifacts/grounded_context_transfer_20261105/check/`. Saved positions and
source hashes were independently checked after execution.
