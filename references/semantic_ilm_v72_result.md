# Continuous image writing fails the complete-answer audit

The completed 38,651,649-parameter writer produces no exact Chinese answers
on the 128 fixed development questions. Foreground F1 is 29.386%, versus
21.873% when question ink is erased. Both questions are correct in zero of
64 article pairs. Four of the six original Chinese requirements fail.
The candidate is not a usable image-language model.

All 12,976 admitted training images were cached. Production completes
40,000 updates, 640,000 draws and 49 complete corpus passes. Active training
takes 2,510.256 seconds, with 1.155 GiB peak allocation for the writer without
the inherited reader. This excludes 15,428.722 seconds for the full cache
stage, earlier codec learning and external foundation pretraining.

The full autonomous audit generates 384 normal/control answer streams. Its
853.492 seconds include 79.648 seconds setup; peak allocated memory is
18.294 GiB including the inherited reader. All 384 retained outputs receive
independent physical scoring, with 12,976 cached payloads authenticated.
The verifier does not replay the full neural generation.

| Measurement | Normal | Question erased |
| --- | ---: | ---: |
| Chinese exact answers | 0/128 | 0/128 |
| Chinese foreground F1 | 29.386% | 21.873% |
| Chinese learned stopping | 100% | 100% |
| English literal exact reference images | 0/64 | 0/64 |
| English foreground F1 | 14.394% | 7.838% |
| English learned stopping | 93.75% | 100% |

The Chinese question effect is 7.513 F1 points, with a 95% source-article
bootstrap interval [5.727,9.631]. It is resolved but below the required
10-point gain. Exact answers, absolute F1, both-correct pairs and minimum
question gain fail; positive effect interval and stopping pass. English
literal reference agreement does not exhaust open-ended semantic correctness.
Sealed cases remain unopened.

## A structural denoising obstruction

Write the 1,536-dimensional velocity head as v = A h + b. Its final hidden
width is 768, so S = span(A,b) has dimension at most 769. Let Q project onto
the orthogonal complement. Before final LayerNorm, Qv = 0 and the flow obeys
Qz(t) = Qz(0). Those initial Gaussian-noise components cannot be transported
by this vector field in real arithmetic.

For the straight-path velocity target y - epsilon, with independent standard
Gaussian epsilon, the per-coordinate expected squared error is bounded below
by

$$
  \frac{\mathbb E\|Qy\|^2 + 1536-\dim S}{1536}.
$$

On the authenticated trained head, the BF16-rounded affine span has rank 769:
767 directions remain. The noise-only term is 0.499349. The first 128 sorted
authenticated training records, containing 3,619 blocks and weighted equally
by example, add target orthogonal energy 0.150753. The resulting
real-arithmetic velocity-loss floor on that sample is 0.650102.

An independent QR-based trajectory check uses the actual trained flow,
the first cached prompt, empty writing history, seed 20267300 and all
32 midpoint steps. Orthogonal-noise RMS stays 0.699684821 to 0.699684823 in
CPU FP32, with drift RMS 1.10e-7. CPU BF16 gives 0.699627645 to 0.699614454,
with drift 0.0002073. Elementwise rounding creates small departures from the
exact subspace, but does not remove this measured obstruction.

This diagnosis applies before final normalization. It is a post-hoc
architectural and CPU arithmetic check, not a replay of GPU answer generation
or a proof that repairing the head will recover language semantics.
The [full-dimensional glyph control](full_rank_flow_v73_control_protocol.md)
tests a necessary generation prerequisite before further language training.

## Source-bound evidence

Raw run: artifacts/semantic_ilm_v72_20261105/.

- Final checkpoint SHA-256:
  20e0c9289182580fd0062d272e7bc470476b05fa4c105b9ece84a2982fe8ce87.
- Generation report SHA-256:
  4490d6a2e56ee3b874ad0ba4c88bd2c692068e99398f8b12ebac8f5278fd8de7.
- Independent verification SHA-256:
  1aa5893eb29a67c7e5b0d3f38a868e9a922cc5cefc2bb8d399cff8a577c58db6.

The standalone CPU diagnosis can be reproduced with
scripts/diagnose_flow_rank_v72.py and a fresh output directory.
The registered V72 language rejection and the general ILM objective remain
unchanged by the diagnostic.
