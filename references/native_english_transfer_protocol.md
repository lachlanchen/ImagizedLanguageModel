# Prospective English passage-reading diagnostic

Specified while the full-page Chinese alignment run is active, before its
after-learning readouts. Keep that run's optimizer, curriculum and three
registered assessment cohorts unchanged. This is a separate evaluation-only
diagnostic, not new training data or a replacement for failed Chinese QA.

The current local optimizer uses Chinese natural questions and Chinese/English
copying. Prepare 32 English natural questions in sixteen same-passage pairs
from sixteen distinct SQuAD 1.1 development articles. The source is the
[author's release](https://rajpurkar.github.io/SQuAD-explorer/), distributed
under CC-BY-SA-4.0; cite Rajpurkar, Zhang, Lopyrev and Liang,
[EMNLP 2016](https://arxiv.org/abs/1606.05250v3). Pin the official GitHub file at
revision `240e165ab706d95bd4323653bb92421f446208cf`, Git blob
`e9a3f913ad1468ebe105b891334ca7b0bc0e2510`, 4,854,279 bytes. Reuse a verified
local file if present. Source weights may have seen this public benchmark;
this cannot be called a pristine test of foundation pretraining.

Choose candidate passages by SHA256 of seed 20269321, article title and full
context hash. Use at most one pair per article. Within each paragraph, consider
questions by source ID and take the first pair with distinct wording,
nonoverlapping annotated answer spans and disjoint answer sets. Validate every
retained reference offset against the complete original passage. Keep all valid
annotated answer alternatives for assessment. No model score enters selection.

Render the entire passage and question at the existing Noto Sans CJK SC face,
32 px, on a 1024-square white page. Keep words intact and retain all source
characters. Reject missing glyphs, unsupported controls or overflowing pages;
never crop around an answer. Render the primary reference separately at
512x256 and require it to fit; record geometry rejection counts. This is a
geometry-limited diagnostic subset, not an official SQuAD evaluation.

For each pair, retain two normal pages, one shared question-erased page and
two passage-erased pages. Question erasure removes the question paragraph and
constant answer marker, preserving every context pixel. The two erased
question pages must be identical. Passage erasure removes just the passage
band, preserving the complete question and all pixels outside that band.
Retain all original coordinates; never move the question upward. Erasure is
based on paragraph layout, not gold answer locations. There are 80 calls per
model: 32 normal, 16 shared question-erased and 32 passage-erased.

Use latent seed 20269421+p and readout seed 20269521+p for passage p, shared
across all five conditions of its pair. Inputs contain only pixels and fixed
generic controls. Keep answer references in a separate file. After the current
controller and all its registered assessments are terminal, compare the frozen
predecessor with the completed alignment checkpoint on the same input-only
index, native top-one decoding and 96-decision cap. Do not load another GPU
model while training is live. Do not change the current protocol after seeing
this diagnostic's results.

Report exact stopped answers against any original annotated alternative,
both-correct pairs, changes under question erasure and correct-answer retention
under passage erasure. Apply NFC and collapsed English whitespace, preserving
case, punctuation and written content. Keep all raw responses and alternatives;
inspect correct paraphrases separately. Correct answers retained without the
passage do not establish use of that passage. Different wrong answers do not
establish understanding. English transfer cannot qualify Chinese understanding,
direct image writing, historical forms or the complete ILM goal.

After selection, calibrate the original source pages and primary answer images
with single-thread CPU Tesseract English OCR. Retain raw recognition, exact
canonical strings and order-sensitive character error. Do not filter, replace
or repair any selected case using OCR. This is input/readability calibration,
never student conditioning or language-quality evidence. The selected v2
preparation has the same 112 image files, input/reference indexes and selected
cases as the initial v1 preparation; it records whole-cohort font coverage.
