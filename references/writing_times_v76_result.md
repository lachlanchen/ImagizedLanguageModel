# Better near-clean reconstruction does not repair free answers

The matched writing-head intervention **fails all four prespecified training-fit
criteria**. Both arms still produce **0/32 exact free answers** on the fixed
training questions. Continued logit-normal training reaches
48.726% foreground F1; uniform-time training reaches
45.028%, a difference of
-3.698 points. All answers stop.
The uniform head is rejected; the failed original candidate is not upgraded.

At the constructed target-assisted t=31/32 probe, uniform-time training improves
near-clean first-block F1 to 99.8602%
and exact blocks to 15/32, compared with
99.0658% and
5/32 for continued logit-normal training.
Improved assisted high-signal reconstruction therefore does not establish a
repair of free answer generation. The time schedule alone is not a sufficient
solution under this fixed intervention. This does not isolate every remaining
cause or rule out other objectives or training budgets.

Each arm starts from the same completed full-corpus V75 checkpoint and trains
only the same 36,403,200-parameter head for 4,000 updates, using all 12,976
training records and 64,000 draws. All causal-condition and stopping parameters
are unchanged. Nine checkpoints of selected examples, positions, noise and
standard-normal time variates match exactly. Both arms sample
700,759 writing blocks. The original law realizes
0 times at or above 31/32;
the uniform law realizes 22,067.
Exposure and effective loss weighting both change with the law.

Training takes 157.554 and
153.230 active seconds, respectively, excluding
earlier codec, reader, feature preparation and full-corpus writer training.
Evaluation reuses the 32 previously verified prompt-image memories; it makes
no new inherited-reader calls. Free generation receives no target or length,
uses the original 32-step sampler, re-encodes emitted pixels and retains every
output. The near-clean probe explicitly receives noisy target codes and gets
no autonomous credit. These are training cases, not unseen language validation.

Independent CPU scoring reconstructs original references, authenticates all
64 free streams, re-decodes retained near-clean predictions and validates
frozen-body hashes and common-random-number traces. It does not replay the
whole neural sampler. Both failed outcomes and the first two fixed raw
answers per arm are retained without output repair.

See the [registered protocol](writing_time_coverage_v76_protocol.md) and
[source-bound receipt](evidence/writing_times_v76_result.json). Full raw
predictions and head/optimizer states remain under
artifacts/writing_times_v76_20261106/. The general ILM goal remains unachieved.
