# Final normal-answer cohort: language learning remains unvalidated

After the registered 128-update Chinese QA run, the final model produces
**0/32 seen and 0/32 held-out OCR-exact answers**, using either the canonical
answer or any answer accepted by the original source. All 64 normal-input
generations are included. The [machine record](evidence/native_paired_qa_final_normal_cohort.json)
binds each input, seed, output PNG and actual decoder tensor to this cohort.

The training set contains 32 questions in sixteen passage pairs, presented
four times. The 32 held-out questions belong to sixteen different passages.
Training completed all 128 updates, changed all 206 adapter tensors and
reduced its objective. The [training verification](evidence/native_paired_qa_training_completed.json)
records those optimization facts. They have not translated into reliable
answer generation, including on the seen questions.

All 64 original normal-answer pages were reviewed. They contain malformed,
often faint passage/question fragments and extra writing, rather than
consistent concise answers. Every PNG was also checked against its saved
actual decoder tensor. No output repair or transcription correction is used.
OCR is imperfect: the unchanged reference-image calibration recognizes
23/32 seen and 26/32 held-out canonical answers exactly. Accordingly, the
zero scores are OCR observations, not an assertion of perfect manual scoring.
The complete raw-page review independently fails to establish useful answering.

This is the completed final **normal-input cohort**, not the complete
initial/final comparison. Initial-model, erased-input and copying assessment
continues under the fixed protocol. No comparative improvement, semantic
understanding or efficiency advantage is established by this partial result.

A separate [native visual flow-learning implementation](native_visual_flow_learning_protocol.md)
is now committed. Its [CPU preparation](evidence/native_visual_flow_training_preparation.json)
includes 384 training image pairs, 64 held-out pairs and 192 fixed assessment
inputs per model. Its full-model training and language quality remain
unverified. Existing language and image pretraining is retained in that
candidate and must be counted when evaluating its capabilities and cost.
