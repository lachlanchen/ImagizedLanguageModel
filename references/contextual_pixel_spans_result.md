# Exact new copying, weak held-out question answering

The completed contextual pixel-span component produces **32/32 exact new
copies** and **32/32 exact seen-question answers**, but only **9/64 held-out
Chinese answers** and **1/32 both-correct paragraph pairs**. Three of its four
prospective requirements fail. It is not a qualified reader or a free
image-language generator.

| Fixed image assessment | Exact source-rendered answers | Unordered column F1 |
| --- | ---: | ---: |
| New Chinese copies | 16/16 | 100% |
| New English copies | 16/16 | 100% |
| Seen Chinese QA, first 32 training records | 32/32 | 100% |
| Held-out Chinese QA | 9/64 | 36.114% |
| Same questions, question ink erased | 2/64 | 18.868% |
| Same questions, passage ink erased | 0/64 | 9.422% |

The normal-minus-question-erased exact gain is 7/64, or 10.938 percentage
points, below the declared 25-point requirement. New copying passes;
natural answers, both-correct pairs and question dependence fail. Unordered
column F1 does not measure answer order or semantic correctness. Development
is repeatedly used public CMRC data, not an independent sealed test.

## Actual model and image path

The model encodes the complete writing image through the calibrated Qwen3-VL
body, without a language vocabulary module, tokenizer, OCR or text output
head. A new **2,378,370-parameter** head scores dynamic spans of ink columns
using contextual visual features and final numeric task-frame states. It
copies the selected original columns into the answer PNG. Candidate units
can be radicals or parts of letters; they are not a persistent glyph alphabet.
Duplicate complete images have their probability masses summed.

The full runtime retains **7,531,277,171 parameters** and externally pretrained
language/vision weights. The small trained head is not a small complete model
trained from scratch. This is a source-copy distribution: it cannot write an
answer absent from the input. No claim of free glyph synthesis, historical
semantics, Qwen parity or efficiency at matched useful quality follows.

All 320 training IDs appear 80 times, for 25,600 draws and 3,200 updates.
Training takes **37.689 seconds**, peaking at **0.316 GiB allocated**, after
source features have been cached. Those costs exclude the encoder: its setup
and four source checks take **139.548 seconds**, and all 544 encodings take
**637.448 seconds** at **18.031 GiB peak allocated**. No source weights are
downloaded or duplicated. Four V77 memories replay exactly before new calls.

Four additional standalone PNG calls load the complete runtime and reproduce
the cached visual descriptors, task memory, geometry and units bit-for-bit,
as well as every answer pixel. They include the first two development
questions and first Chinese/English copies, retaining the wrong answer too.
Source setup takes **78.726 seconds**; the four complete image-to-image calls
take **5.198 seconds** afterwards, peaking at **17.948 GiB allocated**. The
17.289-second main assessment uses cached features and is not reported as
full-runtime inference speed.

Final head checkpoint SHA256:
`85369b1c1547f6fd02e5cc7ea601e202bb03ed8360e6e4964e94b5beec476a91`.
Local numeric checkpoint:
`artifacts/contextual_pixel_spans_v1/training/checkpoint.pt`.

## Actual failures

For the first fixed held-out question, the reference is `国公、白虎`. The
component instead selects `为辐鳍鱼纲鲉形目鲉亚目，旧属鲉科`. For the next
question about ocean distribution it selects the correct `印度洋及西太平洋区`.
Other failures select a related name, extend beyond a correct prefix, or cut
through a character. Anonymous ink units do not guarantee correct character
or answer boundaries. Clean new-string copying does not qualify these answers.

The [complete raw gallery](evidence/contextual_pixel_spans_result/README.md)
and [machine-readable export](evidence/contextual_pixel_spans_result/result.json)
retain every output, source input, reference alternative, chosen boundary,
optimizer update and native replay. Independent PNG scoring reproduces all
256 outcomes and reconstructs every output directly from its declared source
columns. All 544 feature files authenticate. Training and evaluation article
titles do not overlap; all 160 selected paragraph pairs have distinct answer
images. One training pair is excluded for source ink exceeding the font's
vertical canvas. All 128 erasure controls retain no partially erased ink runs.
These checks establish execution and data fidelity, not language competence.

## Consequence for ILM

This supplies a reproducible exact copying component and a genuine seen-data
fit, while rejecting the frozen-context span head as a qualified natural
reader at the registered budget. The contrast identifies a generalization
problem in answer selection and boundaries after copying has become accurate
on its tests. It does not identify a unique cause or demonstrate that a
larger head or more updates will repair it.

There is no matched comparison with V58: source body, physical units,
training data, pretraining, budget and assessment differ. The 9/64 result
must not be presented as a controlled improvement over that model. Further
work must test grounded semantic learning while preserving accurate copying,
and must still demonstrate independent free image generation. The complete
ILM goal remains active and unachieved.

[Prospective protocol](contextual_pixel_spans_protocol.md) ·
[Training and native-image commands](../docs/contextual-pixel-copying.md).
CMRC 2018: Cui et al. (2019), CC-BY-SA-4.0. Qwen3-VL: Alibaba, Apache-2.0.
