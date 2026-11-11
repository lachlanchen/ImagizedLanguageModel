# Next decision: test conditional writing before expanding training

The completed motor trial gives 7/32 exact proportional answers, loses one
previous exact answer and produces no credited book continuation. Its
registered pilot fails. All 216 initial contextual states are identical to
the matched parent. The intervention therefore changes writing and subsequent
pixel feedback; it does not improve the initial reader. Finite fitting loss
does not show whether the new conditional motor can reliably write its own
training targets. That is the next unresolved measurement.

Use the existing final EMA, parent encoder/decoder and authenticated training
state cache. Make no optimizer update, new teacher call or contextual-body
forward. Diagnose one glyph at a time:

1. Select 32 first and 32 later training positions by a fixed hash order of
   their original pool indices, using the prefix `motor-capacity-v1`. Commit
   their identities before generating diagnostic glyphs. Keep every position.
2. From each saved actual training state, sample one glyph using the registered
   32-interval sampler and an index-bound seed. Compare its raw pixels and
   motor coordinates with the supplied target. Run the unchanged parent writer
   from the same state and directly reconstruct the supplied target through
   the same neural codec. Report all three routes.
3. For all 32 normal QA inputs in each layout, reuse the candidate's retained
   first images and saved states. Compare them with the canonical first form
   and the already saved parent outputs. Do not resample. Separate cases where
   native helper wording begins differently: canonical bitmap mismatch is
   not character or semantic error. Reuse the completed target-coverage audit.
4. Preserve all 128 diagnostic positions, physical outputs, per-position scores
   and stopping observations. Do not alter any frozen complete-answer reading
   or score. Inspect the whole diagnostic set before choosing another fit.

Report foreground F1, exact bitmap counts and normalized motor error separately
for first/later training states and fixed/proportional development inputs.
For either training pool, the diagnostic writing gate is median foreground
F1 at least 0.99 and at least 30/32 samples with F1 at least 0.95. This is a
training-state capability check, not a language or generalization gate. These
thresholds are fixed before diagnostic sampling.

If training-state writing fails, first resolve conditional motor fitting or
target geometry; merely adding books is not a supported remedy. If training
writing passes but development writing fails after accounting for alternate
prefixes, test conditional transfer before a larger fit. If early writing is
reliable but complete streams fail, isolate generated-history and stopping
against correct-history execution. No outcome permits a language claim from
one-glyph fidelity. Broader acquisition still needs substantial complete
sequences, autonomous answers and untouched source-group confirmation.

Keep one GPU 0 worker, the project mutex, 48 GiB entry and 24 GiB runtime host
reserves, at most 8 GiB allocated device memory, a 600-second stage budget and
120 seconds of pressure waiting within that budget. The full inherited reader
is unnecessary for this cached-state diagnosis. Preserve both failed motor
setup attempts, both cold replay attempts, all 216 streams and the verified
manuscript. Update the paper's result figures after this bounded diagnosis;
do not trade model investigation for further cover or layout work.
