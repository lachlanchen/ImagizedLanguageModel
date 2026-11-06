# Full-corpus clean writing fails the original Chinese requirements

The completed clean writer produces **2/128 exact
Chinese answer images**, with **37.631% foreground
F1**, versus 23.609% with question ink erased.
Both questions are correct in 0/64
article pairs. The fixed Chinese development decision is **fail**.

The mean question effect is 14.023 F1 points, with 95%
article-bootstrap interval [10.861, 17.487].
Learned stopping is 100.000%. Failed requirements:
chinese_exact, chinese_foreground_f1, both_correct_pairs.

The writer has 64,520,193 trainable parameters and
starts afresh on all 12,976 admitted examples. It completes 40,000 updates,
640,000 draws and 49 full corpus passes in
2589.452 active seconds, at
1.544 GiB allocated memory without the
inherited reader. The 384 autonomous normal/control generations take
679.857 seconds, including 72.053
seconds setup, at 18.390 GiB including the reader.
These costs exclude earlier writing-codec learning, the reused image-feature
cache's original computation and external foundation pretraining.

English literal reference F1 is 16.098%, with
0/64 exact reference images. Those references do
not exhaust valid English answers. The benchmark is reused development data;
this is not a fresh confirmatory or general-language qualification.

Independent CPU verification authenticates all 384 retained outputs and the
full 12,976-example cache, reconstructs source inputs, and independently
recomputes physical scores and all original Chinese decisions. It does not
replay the entire neural sampler or every feedback encoding. Sealed examples
remain unopened. The separately prepared copy probe has its own result.

The first two fixed original references, normal/control input images and
unmodified answer PNGs are exported under evidence/clean_corpus_v75_raw/.
Empty answers have no placeholder PNG. No output marks are repaired, removed
or selected for appearance. All 384 raw streams and the checkpoint remain in
artifacts/clean_corpus_v75_20261106.

The [evidence receipt](evidence/clean_corpus_v75_result.json) and
[fixed protocol](clean_corpus_v75_protocol.md) bind the numerical claims and
their scope. The small V74 training-set fit remains a distinct control.
The general usable ILM objective remains unachieved.
