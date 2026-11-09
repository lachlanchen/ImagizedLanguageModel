# Longer optimization control for the unchanged physical-pixel head

The completed correct-prefix diagnosis finds only 76.831% Chinese and 90.796%
English training-piece accuracy after 1,200 updates. Complete training blocks
remain 0/256 Chinese and 12/256 English even with correct preceding target
pieces. Generated-prefix drift therefore cannot be the sole explanation.
This evidence motivates a bounded fitting test before changing architecture.

Restore the exact original **untrained** 3.453M-parameter head, not its adapted
checkpoint. Retain all 512 original training examples, all physical target
pixels, normalization fitted only on training, unchanged causal architecture
and the same erased/development cohorts. The first 75 training permutations
must match the shorter execution exactly. Run 1,000 complete epochs: 16,000
logical updates, batch 32, 512,000 retained example draws. The extra registered
update-three replay adds one executed optimizer call and 32 draws with no
extra retained learning exposure.

Use the same FP32 AdamW, betas 0.9/0.95, decay 0.01, clipping at one, seed
202611091 and 32-update warmup. Peak rate remains 3e-4 and final rate 3e-5;
cosine decay spans the longer schedule. Thus the intervention changes both
update budget and the timing of learning-rate decay. Do not attribute a
difference to update count alone. Record matched full-training-set NLL at
updates 0, 1,200, 4,000, 8,000 and 16,000, without early stopping or checkpoint
selection. Only update 16,000 receives the autonomous quality assessment.

Repeat the original exact packing/causality/cache checks and actual update-two
checkpoint/update-three replay. Generate and independently score all 896
normal/erased first blocks with the existing auditor, then inspect all six
complete native-pixel plates. Keep the original prospective component gates:
at least 90% exact training blocks in each language, at least 50% exact normal
development blocks in each language and a positive Chinese question-effect
lower confidence bound. Report failures without retuning those criteria.

This is a fitting control on reused development data. Training memorization,
first-block success or a decreasing supplied-prefix likelihood cannot qualify
general language understanding, full-answer generation, stopping or a complete
PNG-input model. The source reader's native QA performance does not transfer
to these custom-interface cached states by assertion. No teacher or lexical
answer generator enters the head's autonomous inference.

Commit the runner and protocol before CPU preparation; bind the original
initial checkpoint, cache, first diagnosis and complete schedule before GPU
execution. Use the existing PyTorch 2.3/torchvision 0.18 environment. CUDA is
hidden for preparation and audit. Execution uses the project GPU0 lock and
the existing small-head guard (24 GiB RAM available, at most 75% swap, 4 GiB
GPU0 free). Preserve GPU1 and other projects. Limit the execution to 900
seconds and 4 GiB allocated head memory. Preserve failure without automatic
retry; do not start Office or another project model job during this execution.
