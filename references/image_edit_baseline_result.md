# Image-only editor baseline: executes, but fails the requested answers

The fixed external FLUX.2-klein-4B baseline completes all 40 generations on
one RTX 4090 D. It does not produce a usable ILM. Offline OCR exact match is
**0/12 literal-copy requests and 0/8 Chinese questions**, with zero exact
answers in the matched blank controls. Visual inspection of every output
finds copied instructions/passages and corrupted writing rather than the
requested answer alone. Words appearing inside a copied prompt do not count
as an answer.

The complete [raw export](evidence/image_edit_baseline_result/result.json)
retains every prompt, reference, answer and codec reconstruction. Its fixed
galleries show all cases, with reference/normal/blank columns:
[00–04](evidence/image_edit_baseline_result/all_cases_00_04.png),
[05–09](evidence/image_edit_baseline_result/all_cases_05_09.png),
[10–14](evidence/image_edit_baseline_result/all_cases_10_14.png),
[15–19](evidence/image_edit_baseline_result/all_cases_15_19.png).
No output is selected by quality, cropped into an answer, repaired with OCR,
or replaced with font rendering.

The [pre-registered protocol](image_edit_baseline_protocol.md) uses source
revision `e7b7dc27f91deacad38e78976d1f2b499d76a294`, four distilled steps,
guidance one, 1024-square prompt pages and a 512-square output canvas. One
generic condition is compiled offline using the verified shared Z-Image
Qwen3-4B encoder. Every sample receives the same numeric condition; all
changing instructions and passages enter through image pixels. This hybrid
conditioner is not claimed to be byte-identical to the stock FLUX text encoder.
That composition limits interpretation of the negative result.

The fresh generator loads no tokenizer or text encoder and checks its actual
module tree for discrete embedding tables. The generator has 3,875,544,576
transformer parameters plus an 84,046,115-parameter codec. Generation performs
no optimizer updates. Runtime retains no OCR, source-answer image, font
lookup or teacher call. The model and conditioner have external pretraining;
this is not a new foundation model trained locally.

| Measured stage | Wall time (s) | Peak allocated GPU memory (GiB) |
| --- | ---: | ---: |
| One-time generic-condition preparation | 17.677 | 8.078 |
| Codec observations: 60 pages, 41 unique calls | 16.459 | 2.580 |
| Complete 40-output generation invocation | 91.656 | 8.943 |

The generation invocation includes setup; no language-quality efficiency
advantage follows from execution speed. Network acquisition took 1,608.062 s
and is separate from those model stages. The offline conditioner has
4,022,468,096 parameters and exits before the generator process starts.

Codec mean thresholded foreground F1 is 97.575% for the 20 prompt pages,
96.907% for the 20 references and 100% for blanks. The lowest reference is
95.564%. Inspection of all reference reconstructions and two input examples,
including the long Chinese/Vietnamese passage, finds the writing legible.
This is supplied-image reconstruction, not autonomous copying or answering.
Byte-identical pages share one reconstruction observation; duplicate GPU
calls are not claimed.

The separate CPU Tesseract evaluator uses fixed `chi_sim+eng`, PSM 6 and NFC
normalization. English whitespace is collapsed; Chinese/numeric whitespace
is removed, with punctuation and case retained. Before any generated-image
scoring it recognizes 18/20 reference pages exactly. It misreads `杜英翰`
and one character in `中侏罗世到早白垩世`. Therefore the reported OCR scores
are exploratory observations, not a calibrated universal semantic metric.
All original outputs remain visible for review.

CPU export checks reproduce each answer PNG from its retained raw decoded
RGB using the actual BF16 postprocess convention. Every codec PNG is likewise
checked, and all 60 codec F1 values are independently reproduced from pixels.
This authenticates saved output and scoring; it does not independently rerun
the transformer, sampler or VAE neural computation.

The result rejects zero-shot use of this fixed-condition composition as an
answering system. It does not show whether paired-image training can learn
the task. That next experiment must preserve an autonomous image-input/output
path and demonstrate actual learning, new-string copying and held-out
prompt-dependent answers. No useful or efficient general ILM is established.
