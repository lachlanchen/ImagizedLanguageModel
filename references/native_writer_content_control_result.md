# Supplied content does not qualify the current writing path

The native image writer remains unreliable even when the intended answer text
is supplied through its original text-conditioning path. The completed
[registered control](native_writer_content_control_protocol.md) produces the
following exact OCR counts on four fixed strings:

| Writer and guidance | Chinese | English | Combined |
| --- | ---: | ---: | ---: |
| Released source, guidance 1 | 0/2 | 0/2 | 0/4 |
| Released source, guidance 4 | 0/2 | 1/2 | 1/4 |
| Adapted writer, guidance 1 | 0/2 | 0/2 | 0/4 |
| Adapted writer, guidance 4 | 0/2 | 0/2 | 0/4 |

All four original clean target pages pass the same fixed OCR rule. The one
exact supplied-content page is the released source at guidance four writing
`garden silver open blue`. Its matched adapted counterpart is incorrect.
These are repeated development cases under a fixed prompt and seed, not
independent benchmark observations. The result does not establish a general
adaptation effect from one changed exact count.

The two Chinese strings are the canonical answers to the previously unfitted
seen question pair. The English strings are one seen and one excluded copying
case. Every supplied-content generation receives the complete original input
page plus the correct answer string, verbatim, in a registered instruction.
Target PNGs do not enter generation. This deliberately tests an offline text
conditioning control and cannot establish independent image-language ability.
The [registration](evidence/native_writer_content_control_registration.json)
contains all four strings, prompt bytes, seeds, image hashes and reference OCR.

Both arms retain the original reader queries and native lexical input table,
tokenizer and unused output head. The source arm uses zero-effect adapters;
the adapted arm loads the exact completed 1,536-update connector/writer
adapters. The failed 128-update reader-query fit is not loaded. The control
therefore retains 6,862,609,631 runtime parameters, rather than the smaller
6,240,279,775-parameter vocabulary-removed ILM path. Neither is a foundation
model trained from scratch in this experiment.

The attempt completes on one RTX 4090 D in 794.092 seconds at 16.409 GiB peak
allocated VRAM, including source loading and all eighteen native draws.
Sixteen draws use supplied content; two are fixed-task execution replays.
Both replays match their original reader inputs, actual decoder tensors and
PNG pixels exactly. The fixed fifty-step schedule accounts for 900 denoiser
calls and 1,300 denoiser batch elements. No optimizer update is performed.

The [independent CPU audit](evidence/native_writer_content_control_result/report.json)
authenticates every original PNG against its saved decoder tensor, recomputes
all exact-match counts from the unchanged raw OCR, verifies both replays and
checks the complete registered generation order. Reader-input tensors for each
supplied string are bit-exact across guidance settings and adapter arms, as
expected when only downstream connector/writer adapters differ. All registered
source, data and preparation bindings remain unchanged.

All eighteen complete generated pages and four original targets are actually
inspected at full resolution. The [raw review](evidence/native_writer_content_control_result/raw_review.json)
records malformed Chinese-like forms, English misspellings, merged words,
unwanted punctuation and instruction text appearing in an output. Source and
OCR access were available, so no blinded review or separate quantitative
manual accuracy is claimed. Complete original
[generated pages](evidence/native_writer_content_control_result/images/),
[inputs](evidence/native_writer_content_control_result/inputs/),
[references](evidence/native_writer_content_control_result/references/),
[OCR assessment](evidence/native_writer_content_control_result/assessment.json)
and [archive hashes](evidence/native_writer_content_control_result/provenance.json)
are retained without repair or reroll. Both model and coordinator processes
terminate and GPU0 is released; GPU1 is preserved.

No arm/guidance cell meets the registered four-of-four supplied-content writing
control. This weakens the premise that correct content conditioning alone is
already enough for reliable output in the tested native editing path. It does
not identify a unique cause: prompt interpretation, reference-image interference
and the source model's writing capability remain possible limitations. In
particular, this experiment does not prove that the released writer cannot
write Chinese under any prompt or sampling setup.

Another larger run of the unchanged query-only recipe is not supported by
these results. The next bounded isolation should compare supplied-content
writing with and without a reference image, under a separately fixed protocol.
That can test the image-editing route before choosing further reader-to-writer
training. It would still be a component control, not a substitute for the
unachieved goal of meaningful answers generated directly from glyph images.
