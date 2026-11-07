# Chinese reading after the native-geometry finding

Previous turn classification: progress. The matched English interface test
recovered 11/32 exact source answers at 432 versus 0/32 at 1024; the existing
adapters reached only 5/32. This protocol fixes the next diagnostic before
model calls. It does not admit another optimizer or qualify image answers.

Use all sixteen existing held-out CMRC passage pairs: 32 normal questions and
sixteen shared question-erased pages. Preserve their original ordering,
distinct-answer references and per-passage latent/readout seeds. The original
input index SHA256 is
`4994faea12d72f6522630ec370fa9355b23ccceb84060dc43829f2437d6e77e6`;
references SHA256 is
`a498fb556746e4a8c90e9d1995f2249e578d612424efe3bf92edc75cabffed3f`.
The original source remains the existing CMRC short-passage preparation,
SHA256 `742a473db53d1a55e319f0b52befc8da5fa3d884df395b37e66581e882d1496c`.

Run three frozen-source arms, each with all 48 conditions:

1. Original complete 1024-square image pixels.
2. Whole-square bicubic resize to native 432, with no crop or reflow.
3. Original visible source text, reconstructed and hash-checked against the
   original page manifest. Erased text contains only the remaining passage,
   without the question or answer marker. This is a privileged text-input
   capability control, not an image-only learner or teacher used at inference.

Reuse the pinned authenticated Show-o2 HQ weights, BF16 body, sampled FP32
native Wan codec, cached top-one decoding and 96-decision cap. Use the same
generic image instruction as the English check. The text-only arm uses the
existing native text-control chat wrapper and its generic request, with the
same per-passage readout seeds. Do not supply reference answers, correct answer
history, OCR, another model or learned adapters. Run no optimizer updates.

Save all 32 normal 432 codec reconstructions from the exact latents used for
reading. Independently recognize all 32 original pages, all 32 resized pages
and all 32 reconstructions with Tesseract `chi_sim+eng`, PSM 6 and one CPU
thread. Compare complete visible source transcripts by NFC/Chinese whitespace
normalization and Levenshtein character error. Recognition never selects cases
or conditions the native model. Preserve each input/reconstruction and every
raw response, including cap failures.

Use the unchanged paired-question scorer: exact stopped normal answers,
both-correct pairs, different normal outputs, changes after question erasure
and shared erased-page matches. Keep order-sensitive correctness distinct from
character-multiset F1. The text capability control and geometry comparison
must remain separate. A text success with image failure would localize a
modality/interface gap without selecting a repair. A native-image recovery
would justify using that calibrated baseline, not claim new language learning.
Either result still requires independent continuous image-answer generation.

CMRC passage images preserve CC-BY-SA-4.0 attribution to Cui et al.,
EMNLP-IJCNLP 2019. These public, repeatedly inspected development cases are
not a sealed benchmark or a replication of the source authors' evaluation.
