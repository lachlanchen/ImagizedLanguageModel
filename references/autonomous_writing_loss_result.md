# The recognition loss is weak on assisted predictions and incomplete on real pages

The frozen training reader scores every one of the completed image writer's
64 actual autonomous outputs. Mean gold-answer NLL is **1.041635 nats on seen
examples** and **0.574959 on held-out examples**, compared with .033706 and
.043822 on their clean reference pages. Mean recognition loss in the earlier
255 assisted training updates was .031519. Those training updates use noisy
correct target latents and changing checkpoint states; the free outputs use
the final state and independent noise. These observations identify a gap,
not a controlled decomposition of its cause.

The [complete report](evidence/autonomous_writing_loss_result/report.json)
retains all 64 raw transcripts, expected strings, target NLLs and output
pixel hashes. The probe checks each PNG against its retained neural tensor;
it never loads or optimizes the writer. The reader uses unrounded actual
postprocessed decoder pixels, not an OCR-repaired or rerendered answer.
The [original outputs](recognition_guided_writing_result.md) remain unchanged.

| Observation | Seen, 32 cases | Held-out, 32 cases |
| --- | ---: | ---: |
| Mean reader NLL, actual autonomous outputs | 1.041635 | .574959 |
| Mean reader NLL, clean references | .033706 | .043822 |
| Training-reader greedy exact transcripts | 14/32 | 19/32 |
| Original separate CPU evaluator exact | 6/32 | 9/32 |

The training reader is **not an independent quality judge**. Its more generous
transcripts do not replace the original scores or qualify a language model.
The complete raw galleries visibly contain failures beyond either recognizer's
transcription errors.

## The reader catches some errors and ignores others

Previously documented visible errors are inspected without selecting a new
successful subset. Training case 004, which omits characters, receives target
NLL 3.271747 versus .000949 on its reference. Case 012 adds a line from the
instruction and receives 1.356088 versus .000877. Case 015's corrupted English
and extra line receive 7.148628 versus .006676. The teacher can penalize these
actual failures; the added training term was applied to a different regime.

It is not a complete content constraint. Validation case 002 visibly repeats a
character, yet the reader returns the expected string and target NLL .000422
(reference .000346). Validation case 026 duplicates its line, yet the reader
again returns the expected string, with NLL .042427 (reference .000192).
The extra marks in validation case 000 similarly disappear from its greedy
transcript. A recognition reward alone can tolerate incorrect visible writing.

## Next learning test

The [prospective autonomous-writing protocol](autonomous_writing_protocol.md)
tests gradients of a recognition plus balanced pixel loss on the actual final
50-step image. The generator receives no target mixture; it starts from
independent noise with only the input image and fixed generic condition.
Reference ink/background supervision addresses the observed reader blind
spots, while limiting this test to a canonical writing style. It is not a
claim of font-invariant or historical-script understanding.

This follows the established differentiable-sampling principle in
[DRaFT, Clark et al.](https://arxiv.org/abs/2309.17400). It is a new local test,
not a newly invented backpropagation algorithm. Exact native sampler replay
and full-trajectory gradient/memory checks precede any learning pilot.
The [input-encoding correction](autonomous_sampler_correction.md) records a
failed numerical preflight without relaxing its bounds. Pilot copying,
held-out semantic answering and broader language learning remain separate
requirements; the general ILM goal is not achieved.

The 64-output reader probe took 29.870 seconds and 3.395 GiB peak allocation
on GPU0. Reproduce with `scripts/probe_autonomous_writing_loss.py`. Source
weights, code hashes, input bindings and the unchanged admission report are
identified in its full report. No additional model weights were acquired.
