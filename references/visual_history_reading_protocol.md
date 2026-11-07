# Can the native visual reader continue a visible answer prefix?

Prospective diagnostic, 8 November 2026. The source-descriptor canvas comparison
fails every exact unseen copying and QA check. Extending that unchanged
four-pass MSE recipe is not admitted. This experiment tests a different
history interface before choosing another image-writing learner.

The completed history audit found only two matched saved native-language
states at 75 intermediate image-block boundaries. Inserting learned image
embeddings into a pretrained language continuation and rereading a page of
actual written pixels are different operations. Test the latter directly:

    H_t = F_native(visual_page(question_pixels, prefix_pixels), fixed_controls)

The upper panel retains every original question pixel. A separator and an
English label identify the lower panel as the answer already written. The
lower panel contains a correctly rendered, possibly empty reference prefix.
No remaining answer appears there. The whole new page receives the existing
declared native image-budget transform. Generation uses the imported source's
native text projection only as an **offline instrument**, with native KV
continuation. No new writer, adapted source or image-answer capability is
claimed. Reference history is supplied, so this is teacher-assisted history,
not an autonomous image rollout.

Use all 44 cases in the authenticated source-helper inventory: twelve copying
requests and 32 seen Chinese QA questions in sixteen different-answer pairs.
For each nonempty canonical answer of n Unicode codepoints, take the sorted
distinct positions `{0, floor(n/2), max(0,n-1), n}`. Every case and position
remains, including terminal and punctuation/whitespace positions. Prefixes are
rendered as complete strings at those boundaries. They are new physical
observations, not asserted pixel-equivalent to the old 64-column histories.
The old source-helper text and images are authenticated before construction.

For every position assess two fixed routes:

1. **Visual page:** question and answer-prefix pixels only, plus fixed numeric
   task controls. The controls ask for only the missing answer continuation,
   without repeating the prefix, and for empty output if complete.
2. **Native text:** the same question and already-written prefix supplied as
   source text. This is a privileged language-capacity control.

At the middle nonterminal position of every case with such a position, assess
one additional visual page whose prefix ink is erased at the same geometry.
Its question and label pixels remain unchanged. It is a counterfactual input;
its comparison target remains the original missing suffix. No erased case is
discarded if the source gives the full answer instead.

Use the frozen shared Qwen3-VL 8B source, unchanged zero-output adapters,
deterministic top-one native decoding, native EOG events and a uniform cap of
64 language pieces. The output projection uses exactly the prior helper's
BF16-rounded weights, held in FP32 to avoid a per-decision cast. Save raw
decoded strings, every selected native ID and score, every pre-decision hidden
state, initial prompt states, input hashes and source identities. No retries,
prompt tuning, repair, selected best sample, teacher training or optimizer
updates. Unicode/tokenizer data and the output head are restricted to this
diagnostic, not deployed as an ILM inference path.

After generation closes, report literal outer-whitespace-trimmed suffix
exactness, first remaining codepoint agreement, stopped suffix exactness,
normal-versus-erased middle-prefix gains, case-level complete success and
terminal empty/stopped behavior. Trim only outer Unicode whitespace; do not
remove quotes, explanations or punctuation or change internal whitespace.
First-codepoint agreement applies to nonempty trimmed suffixes; empty suffixes
remain separately reported rather than being silently dropped. All raw
outputs and conditions stay in the receipt. First codepoint is not a glyph
segmentation rule, image-generation score or semantic-equivalence judgment.

A native-pixel-history writer trial would be admitted only if visual reading
gets at least .90 first-codepoint agreement on the copying nonempty suffixes,
at least .80 on QA nonempty suffixes, at least .80 empty-and-stopped terminal
answers separately for copying and QA, and at least .15 first-codepoint gain
over erased prefixes on both tasks' matched middle positions. These are
practical decision floors, not significance tests, linguistic completeness or
ILM qualification. Negative results distinguish this interface's failure from
an assumed successful native reader. They do not prove that all such layouts
or learned readers must fail. A positive auxiliary readout would still need
an actual image-only writer, unseen assessment and end-to-end verification.

Commit source and controls before calls. Prepare and independently inspect
the physical pages on CPU. One persistent owned GPU0 job acquires the project
lock and materializes shared weights once. Check host memory and project
processes before launch; preserve GPU1's unrelated runtime. Complete each
generation once, journal its atomic result, and resume only missing calls
after authenticating completed outputs. Record setup and measured inference
costs separately; enforce 22 GiB allocated memory and a two-hour wall bound.
Release the exact owned runtime on completion. CPU verification reconstructs
page/prefix/erasure pixels, recompiles numeric controls, checks output decoding
and source-projection argmax from retained states in batches. It does not
claim a second full vision-body replay.

CMRC2018: Cui et al. (2019), CC BY-SA 4.0. Source: Alibaba Qwen3-VL,
Apache-2.0. Existing copying requests are prepared examples. Neither external
pretraining nor successful text decoding constitutes an independent visual
language model. The overall ILM goal remains unachieved.
