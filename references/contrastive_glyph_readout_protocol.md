# Loss-only contrastive readout intervention

The first broad readout compiler is rejected: its synthetic held-out glyphs
mostly become dense composites and genuine-state rendering regresses.
Preserve all measurements and raw images. This follow-up changes one loss
term while retaining the architecture, primitive, original baseline
initialization, cached genuine states, synthetic calibration states, sampler,
seed, 6,000 updates, optimizer/schedule, frozen stop and all resource limits.
It does not initialize from the failed candidate. Reuse calibration by verified
references; do not copy the source weights or call the teacher again.

For readout z, normalized training-only semantic anchor a_c and target y, add

    L_discriminate = -log exp(cos(z,a_y)/0.07) /
                          sum_c exp(cos(z,a_c)/0.07).

Its weight is 1, added to the previous MSE +0.2 cosine discrepancy + balanced
pixel BCE. Every training anchor supplies negatives. The anchors and integer
targets are offline training data and evaluation labels; no anchor table,
nearest-anchor selection or lexical output is deployed. Runtime G and D are
unchanged. This tests discrimination among competing forms, rather than
assuming that sharper rendering follows from lower regression loss.

The same full held-out synthetic/genuine pools and prospective gate apply:
mean physical glyph F1 >=0.95 and at least 80% of glyphs with F1 >=0.95 on both
routes. Class separation during fitting and glyph quality are separate
measurements. Qualifying a teacher-function compiler is not language
understanding. Record failure without running expensive language inference;
retain every raw held-out glyph and cold-replay metrics before accepting it.

No development data enter training and no teacher service or native contextual
body is loaded. The source body and its pretraining remain part of the model's
provenance; these small readout costs cannot be reported as foundation-model
training costs or an efficiency advantage at matched language quality.
