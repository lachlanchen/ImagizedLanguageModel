# Full context is needed by the tested reader; the new writer is unqualified

Restricting the unchanged source helper to the final 52 task positions after
identical image prefill reduces exact copy answers from **8/12 to 0/12** and
exact training answers from **19/32 to 2/32**. All 44 first decisions and all
44 cached BF16 reader memories remain identical. The intervention masks older
keys without deleting them or changing rotary coordinates. Poisoning the
masked keys leaves the next state identical, while the unmasked positive
control changes it. Offline replay confirms all 1,239 source projection choices.

This is a source-decoder visibility experiment using its symbolic head as an
offline diagnostic. It is not a deployed image model result. It rejects the
tested frozen suffix interface; it does not establish a lower bound on learned
compression or prove that the image writer has only one failure. In particular,
the earlier writer also failed to predict clean glyph codes given correct
history and answer length.

The next image model retains all native prompt KV and all 36 inherited contextual
layers. A learned continuous image encoder feeds generated writing back into
that body. A clean image-code predictor and learned structural stop supply the
output. Deployment has no tokenizer, symbolic vocabulary projection, glyph
lookup inventory, OCR or teacher service. The inherited weights were externally
pretrained. The complete model has **7,582,980,210 parameters**, of which
**46,899,713** new interface parameters are trainable; this is not a model trained
from scratch at that cost.

The actual inherited-body execution has zero error against the retained native
reference in its forward check. Input gradients traverse all 36 layers; causal
future exclusion, right-padding and incremental-cache checks pass. Untrained
generation writes two physical image blocks and rereads those emitted pixels.
The execution peak is **16.638 GiB** on one 4090. It contains **zero optimizer
updates** and receives no language credit. The original verifier's checkpoint
metadata error is preserved. The supplemental verifier reproduces both original
GPU feedback hashes and measures CPU/GPU codec error **2.98e-6** against a
prospectively declared tolerance of 1e-4.

The [Chinese training protocol](native_chinese_v80_protocol.md) specifies the
next complete optimization and fixed output audit. The original full-corpus
writer remains a failed model. A usable image-language model, superiority to a
token model and general historical-script understanding remain unproved.

The [machine-readable receipt](evidence/native_context_v80.json) retains all 44
unmodified masked/full helper answers and source hashes. Re-export with
`PYTHONPATH=. python scripts/export_native_context_evidence_v80.py` when the
authenticated local experiment artifacts are present.
