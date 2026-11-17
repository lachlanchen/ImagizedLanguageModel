# The 64-bit global codec reduces source refinement but loses glyph fidelity

The registered comparison is complete and fails six of nine conditions. Both
fresh 2,030,465-parameter codecs finish all 6,000 updates, from identical tensors
and source order. Hard quantization is the only changed computation. The 64-bit
candidate does not qualify the proposed writer or admit conditional language
training. Preserve both final checkpoints, all outputs and every failed gate.

| Physical measurement | Continuous 64-dimensional | Binary 64-bit |
|---|---:|---:|
| Training foreground F1, 64,041 cells | .991586440 | .957150751 |
| Validation foreground F1, 4,907 cells | .932736083 | .810400756 |
| Validation macro-form F1, 952 forms | .906819307 | .785655237 |
| Validation exact cells | 68/4,907 | 0/4,907 |
| Validation macro-form exact rate | .002100840 | 0 |
| Source false blanks / false ink | 0 / 0 | 0 / 0 |
| Mean source correction-tree events | 8.471181 | 22.015026 |

Binary source mean plan-plus-correction work is 30.015026 events, versus the
previous direct tree's 97.302489 on the same complete support: a ratio .308471.
It passes the registered 40% event-reduction condition. This is a physical event
count, not encoded bits, measured throughput or language efficiency. On the
validation cells, its plan-plus-correction mean is 90.105156 events; a matched
direct-tree validation denominator is not measured here. Do not compare that
subset directly with the complete-source mean as a validated speed ratio.

The source average contains repeated common masks. Binary training exactness
is 45,663/64,041 cells, but training macro-form exactness is only .057364700.
Per-form physical results and held-mask reconstruction therefore matter even
when many repeated cells reconstruct exactly. The comparison preserves all
68,948 original cells and their multiplicities. It is not a source-alias audit.

The hard branch loses .122335327 validation F1 against the matched soft branch.
The continuous branch also fails its .98 validation ceiling. Thus hard
quantization worsens this tested recipe, while the common low-dimensional
global codec has a separate generalization limitation. These observations do
not uniquely distinguish capacity, spatial geometry, optimization or data
weighting, and do not prove a 64-bit information-theoretic limit. Do not tune
the failed thresholds, extend unchanged epochs or select a different snapshot.

## Execution and independent recount

Both complete-training-support gradient probes cover all 64,041 training cells,
with finite nonzero bottleneck and decoder gradients. Each actual trial update
changes weights, lowers the measured complete training objective, and resets
exactly before fitting. Both update-three model/optimizer replays are exact.
Each arm retains 768,000 fitting draws; the two full-support update/reset trials
and extra 128-image replay per arm are separately recorded costs.

The [independent CPU audit](../scripts/export_binary_codec_comparison.py)
recounts every one of the 275,792 initial/final reconstructions. It verifies
every XOR correction and its physical event count, all 12,000 retained updates,
the complete actual source order, final checkpoint tensor hashes and original
source bindings. It does not repeat neural fitting or decoding. The
[hash-bound result](evidence/binary_codec_comparison_result_20261118.json)
retains the original neural measurements, full training curves and raw-file
hashes. All input and prior model files remain unchanged.

The fixed [validation](evidence/binary_codec_comparison_20261118/held_fixed_reconstructions.png)
and [training](evidence/binary_codec_comparison_20261118/train_fixed_reconstructions.png)
sheets show references, actual continuous and binary outputs, and binary XOR
corrections. Their source indices were selected by source order, never output
quality. Every displayed pixel is checked against the raw arrays; enlargement
is nearest-neighbour. Inspection shows recognizable structures alongside
missing, displaced and substituted strokes. No transcript or semantic score
is assigned from this unblinded physical review.

The monitored neural comparison takes 256.499 seconds with zero pressure wait,
.593 GiB peak allocation, 2.755 GiB peak process RSS and 51.280 GiB minimum
sampled available host memory. The supervisor records 265.262 seconds; all
owned processes exit and GPU0 is released. Foreign GPU1 remains unchanged.
The CPU audit takes 40.003 seconds and 1.719 GiB peak RSS. Source preparation
is separately recorded. No contextual foundation is loaded or trained; these
costs establish no complete ILM efficiency or matched-quality LLM comparison.

This is supplied-image reconstruction with a physical mask holdout, not language,
unseen works, novel writing styles or historical semantic transfer. The original
complete-language and untouched-source criteria remain unchanged. The general
ILM goal remains active; this global 64-bit compression recipe is rejected.
