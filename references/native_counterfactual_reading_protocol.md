# Prospective swapped-fact reading control

This additional diagnostic is fixed while the state-alignment optimizer is
still live, before its after-readouts. It supplements the declared 160-case
natural-QA/copy assessment and cannot replace it, enter training, change its
six gates, select a checkpoint or complete the image-answer ILM goal.

Sixteen explicitly fictional registers cover eight English and eight Chinese
scenarios. Each register assigns two different objects to two named places.
Two passage versions swap the places while keeping the object names,
instructions and questions unchanged. Randomized initial assignment and
record order use seed 20270521. Both versions contain exactly the same
character multiset. Each question asks for one object's printed location;
the correct answer must change between the two versions. These are synthetic
grounding controls, not a natural-language benchmark or historical facts.

Each scenario has eight independent model inputs: four normal combinations
(two passages by two questions), two shared question-erased pages and two
shared passage-erased pages. Passage-erased pages must be pixel-identical
across versions for each question. Question pixels remain identical across
versions. Question-erased pages must be identical across questions for each
version. Preserve every character and both complete page geometries, using
the existing 32-pixel font, 1024-square source and 432-square derived view.
Do not crop around an answer or replace cases from model/OCR outcomes.

All eight calls within a scenario share latent seed 20270321+i and readout
seed 20270421+i. Each arm therefore has 128 calls: 64 normal, 32 independently
question-erased and 32 independently passage-erased. Assess the frozen native
source, the one final 2560-update learner, and an original-visible-text source
control. No new optimizer runs. The text arm is a privileged language-capacity
control. Changing condition metadata or case IDs must not enter the neural
prompt. Use the existing generic request, BF16 body, FP32 codec, greedy
96-decision cap and unchanged normalization/EOS criteria.

Measure normal correctness, same-question correctness on both passage
versions, both-question correctness within each version, and all four normal
answers correct in a scenario. Report erasure changes and correct normal
answers retained/lost without the passage separately. A model that ignores
the changed context cannot answer the same question correctly on both
versions by a fixed answer. Different outputs alone are not sufficient.

Prepare all 128 input/control and 64 separate reference images, verify mask
invariants and source font coverage on CPU, and recognize all normal source
and reference images with independent one-thread Tesseract calibration. Keep
raw recognition errors without repair or filtering. Do not use OCR as model
conditioning. Input indexes contain only raster paths, hashes, conditions and
seeds; visible transcripts and expected answers remain in separate files.

GPU calls must wait until the current optimizer, complete primary assessment
and additional source-equivalence control are terminal. Record this cohort
separately, preserving all declared primary results whether it succeeds or
fails. Success would show a bounded form of reading newly printed facts,
not generic reasoning, language-model parity or independent image answers.
