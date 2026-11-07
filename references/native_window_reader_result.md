# Native visual windows preserve resolution but do not repair reading

The full-resolution source codec preserves Chinese writing substantially
better than the small-view reconstruction, but two frozen feature-interface
controls still fail natural QA. These outcomes reject their zero-shot use;
they do not establish a working image-language model.

| Same source and fixed questions | Chinese exact /32 | English exact /32 |
| --- | ---: | ---: |
| Original global encoder at 1024 | 0 | 0 |
| Whole-image native 432 resize | 0 | 11 |
| Native semantic windows, full context | 0 | 2 |
| Native semantic windows, pooled context | 0 | 0 |
| Original-text control | 16 | Not measured here |

Chinese/English both-correct passage pairs are 0/16 for both window variants.
Full-context Chinese normal outputs stop on 30/32 cases and change after
question erasure on 31/32. Pooled-context outputs stop on 31/32 and change
on 17/32. Changed output alone does not establish correctness. English
full-context scores 2/32 normal and 0/32 passage-erased answers; both correct
normal answers become incorrect without the passage. Pooled English normal
and passage-erased scores are both zero. Every response is retained.

The full-context path applies the frozen semantic encoder independently to
nine 27-square windows of the original 64-square image-patch grid, averages
overlaps in FP32 and restores full row-major order. All 4096 visual positions
remain in the language prefix, which has 4135 total positions. The pooled
control then adaptively averages fused visual features to 729 positions,
giving the native 768-position prefix. It changes feature granularity and
cannot claim lossless memory preservation. Original source pixels, VAE
sampling, time/role embeddings, source weights, paired seeds, BF16 body and
96-decision cap remain fixed. There are no optimizer steps or new parameters.

Each implementation first passes four native single-window equivalence cases:
prefix and attention masks match exactly, and all eight original/window
readouts replay the retained source decision IDs. This numerical check
qualifies the native-size special case; it does not qualify the new full-page
interface. Each full assessment then executes all 48 Chinese and 80 English
normal/erased conditions. The two controls therefore make 256 full-page
calls and sixteen separate replay calls.

Independent full-resolution Chinese codec reconstruction has **0.086875
character error** on all 32 normal pages, versus **0.705221** at 432. The
original clean-page check is 0.089771. No complete transcript is OCR-exact;
these are recognition-based readability observations, not exact measurements
of retained latent information. Both variants' 32 reconstructed PNG hashes
are identical, verifying that their feature modifications do not change this
codec control. No OCR enters model conditioning or selects cases.

[Window protocol](native_window_reader_protocol.md),
[pooled protocol](native_pooled_reader_protocol.md),
[full-window export](evidence/native_window_reader_result/result.json) and
[pooled export](evidence/native_pooled_reader_result/result.json) retain the
source bindings, all outputs, 64 reconstruction observations and executed
implementations. The exports bind 42 and 43 files; identical PNGs are content-addressed
by Git. Original source images remain bound to prior public evidence.

The window idea is related to using a frozen encoder at its native scale in
[S2](https://arxiv.org/abs/2403.13043v2). This overlapping latent-feature
implementation is not a replication of S2 or a new foundation architecture.
CMRC 2018/SQuAD 1.1 preserve CC-BY-SA-4.0 attribution to Cui et al. (2019)
and Rajpurkar et al. (2016); source/model are NUS Show Lab, Apache-2.0.

The next [state-alignment experiment](native_state_alignment_protocol.md)
preserves the working native low-view source at initialization and learns
only a correction from the fine view. Offline original-text states supply
training supervision. That experiment has no completed language result yet.
It must beat the real native-source reference, including 11/32 English
answers, rather than treating either failed window control as the quality bar.
Direct continuous image-answer generation remains unachieved.
