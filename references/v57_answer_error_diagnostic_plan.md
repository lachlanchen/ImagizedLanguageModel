# V57 answer-structure diagnostic

2026-11-05, planned after the copy arm's primary/blank/pair outputs were saved
and their aggregate scores inspected; the matched no-copy audit was still
running. This is an explicitly post-result diagnostic. It changes no weights,
decode rule, quality gate, primary output or audit decision.

First use saved raw images only. Compare the first emitted image, the longest
aligned correct prefix and exact reference-length agreement with all retained
references. Separate answers whose first image is correct from other answers.
Measure image repetition, stopping/capping and whether a complete correct
reference prefix is followed by unwanted additional output. Report all 256
primary rows, not selected successes. No model or external OCR call is needed.

If this leaves the failure unclear, a later read-only teacher-forced diagnostic
may separate initial answer choice, continuation and terminal-stop likelihood.
That additional diagnostic must remain separate from autonomous generation;
it may expose a gold prefix only to analyze the already trained checkpoint.
No new architecture or training run is authorized by this diagnostic alone.
