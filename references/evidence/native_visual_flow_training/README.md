# Completed native-flow training; language assessment pending

This package retains all 24 original fixed training-preview PNGs at the twelve
registered updates, plus both complete input/reference images. Every preview
is authenticated against its saved actual decoder tensor by the CPU audit.
All 24 original pages have been visually inspected. The final two previews
still give incorrect short answers. No image, transcript or contrast is repaired.

These are seen training examples with fixed monitoring seeds. They are not
held-out results or a checkpoint-selection rule. The complete initial/final
question, erasure and copying assessment is separate and still pending.

The final checkpoint contains 1,536 actual AdamW updates over 384 image pairs,
with 852 adapter tensors and 37,380,992 trainable parameters. The complete
pretrained runtime has 6,240,279,775 parameters. Its inherited language/image
pretraining remains part of the system. Training takes 11,450.839 seconds
(3.18 hours), at 15.697 GiB peak allocated GPU memory. Execution on one card
does not establish foundation training or efficiency at useful language quality.

`training_audit.json` preserves actual optimizer, schedule, RNG and pixel
authentication. `training_report.json` is the original completed report.
`raw_visual_review.json` binds all monitoring-image observations without a
blinded-review or quantitative manual-accuracy claim. `result.json` binds
every published file. The full checkpoint and actual tensors remain local
under the experiment directory; no weights or private runtime logs are copied.
The source data attribution and licence are retained alongside the images.

No new independent neural replay, correct answering, historical-script
understanding or independent ILM qualification is claimed by this package.
