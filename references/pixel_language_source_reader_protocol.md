# Native reader check on the current Chinese development cohort

The [likelihood diagnostic](pixel_language_first_block_likelihood_result.md)
finds transferred paired-question preferences but much lower probability for
correct reference images than for the emitted first images. Before changing
the image readout, check whether the imported reader itself can answer these
questions through the actual visual path.

Use all sixteen Chinese development questions, without selection by prior
quality. For each, run the original normal PNG, its question-erased PNG and
the source passage/question as text. The text condition follows the earlier
V71 helper's structural format; its layout is not identical to a rendered
page. The fixed source task requests a short direct answer. Cap each helper
response at 64 source vocabulary units, record termination and retain every
raw response. These are reused development cases, not a sealed evaluation.

The image routes use the same native image-budget transform, numeric task
vectors, visual/context implementation and shared pretrained weights as the
completed pixel experiment. Each image prefill must reproduce its actual
saved ILM reader-memory fingerprint before any response is interpreted.
No trained writing adapter, learned initial writing query or raster head is
used by this source baseline. Source contextual adapter outputs must be zero.

Source vocabulary projection and previous-symbol embedding lookup are local
helper variables, never attached to the image actor. They intentionally
provide a source-reader baseline. They are not ILM image answers and cannot
serve as a deployed OCR/text/font replacement. Gold answers are used only
after decoding for literal-reference measurements. Report raw text and caps;
literal equality is not a general semantic-accuracy estimate.

CPU preparation binds all original images, their expected fingerprints, the
source text file, task compiler, numeric vectors and committed implementation.
It makes no model calls or weight copies. The GPU stage requires the training
owner to have exited and uses the same global GPU0 lock and bounded
read-only admission as the completed likelihood diagnostic. Enforce the
22 GiB allocation cap and monitor host reserves and memory pressure during
decoding. A pressure spike pauses further reader computation until the same
limits recover, with at most 300 seconds of waiting across the whole job.
Available RAM below 24 GiB causes immediate exit. No other process or kernel
setting is changed. Preserve partial and failed executions rather than
overwriting them; include actual waiting time in the recorded runtime.

The result distinguishes three possibilities: usable native image reading
with a failing image readout, a gap between source text and image reading, or
a source body/task limitation in both modes. None by itself validates the ILM.
Development responses and hidden states must never become training examples.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python scripts/check_pixel_language_source_reader.py \
  --stage prepare --output artifacts/pixel_source_reader_new
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=. \
python scripts/check_pixel_language_source_reader.py \
  --stage run --output artifacts/pixel_source_reader_new
```
