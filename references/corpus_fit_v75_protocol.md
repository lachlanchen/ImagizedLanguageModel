# Locate the clean full-corpus prediction failure

Early V75 development outputs remain malformed despite the earlier fixed
32-question fit. Diagnose the completed full-corpus checkpoint after the
original audit, its CPU verification and the queued copy probe finish.
This is a post-hoc training-set diagnosis, not a new language qualification.

Reuse exactly the 32 CMRC cases already fixed by the V74 paired-learning
receipt. They are included in full-corpus training. Make one actual-image
autonomous call per case with seed 20267420 plus pair index, the ordinary
128-block cap, 32 endpoint updates and learned stopping. Preserve all outputs
and actual reader memory. Only prompt pixels enter these calls.

After each autonomous call, load the original target and its image codes.
Render a supplied-code control through the fixed codec. Also sample one
complete assisted writing stream using the known target length and correct
reference image-code history before every block. Use the same per-case seed
and sequential single-block sampler calls as free generation. This assisted
path receives no autonomous-answer credit. Its metric is pixel agreement
at supplied length; it does not claim learned stopping. Check the first image
and first stop probability against the free path when a free first image
exists, since they share prompt, empty history and noise seed.

After all 32 actual-image calls, probe the first-block clean predictor at
times 0, 0.1, 0.3, 0.5 and 0.9. Use one fresh Gaussian noise vector per case
from seed 20267597, common across times and correct/paired conditions.
Compare the owning prompt's initial condition against the other question's
condition from the same visible-passage pair, holding target and noisy input
fixed. Retain both predicted continuous codes and recompute their MSE on CPU.
This identifies dependence on question conditioning relative to visible target
signal; it is not semantic accuracy or a causal attribution of training failure.

Report free, assisted and supplied-code physical scores for every case,
first-block scores, initial-path agreement, reader-cache equality and latent
errors by time. Keep all failed outcomes. The source cohort, number of updates
and exposure differ from the small V74 fit, so no single-factor causal claim
is supported. No optimization, original-gate change or sealed access occurs.

Independent CPU verification authenticates all inputs and raw files,
reconstructs source input layout, re-renders the frozen supplied-code decoder,
and independently recomputes physical and latent-error summaries. It does not
replay the full neural sampler, inherited reader or every feedback encoding.
Use the shared project GPU lock and physical GPU 0; create no GUI stack.
