# Native contextual continuation with image input and output adapters

V77 identifies answer content before the failed small writer: source image
decoding gives19/32 exact training answers and8/12 copy answers, while the
image writer gives zero. The identical first native state supports an owning-
only answer-prefix projection on22/32 training cases. V78 then masks direct
access to older visual keys, keeping positions fixed: copy exactness falls
to0/12 and training exactness to2/32. The frozen decoder does not retain its
answers using the suffix alone. This does not prove an information-theoretic
lower bound on learned compression, but it rejects assuming that suffix is a
sufficient drop-in interface for the current decoder.

The successor retains the complete prompt-derived native KV and its pretrained
contextual dynamics. It learns image-history input embeddings, a continuous
image-code output density, and structural stopping. It has no fresh small
language transformer, tokenizer, Unicode lookup, language vocabulary head or
deployed teacher. Existing source weights remain attributed external pretraining.
This is an implemented candidate, not a language-success result.

Let a prompt-only frozen reader produce layerwise key/value pairs P, the final
state h0 and a rotary offset. A frozen writing codec encodes a32-by64 image
block as two continuous768-vectors. A shared learned two-layer image encoder
maps each tile code to a4096-dimensional contextual input embedding. New image
observations use ordinary ordered structural positions, with inherited rotary
coordinates continued from the prompt. The frozen contextual body attends to
all real prompt keys and earlier image observations. It yields states h1,h2,...
after completed image blocks. The state before block j never sees that block
or later image observations.

A full-dimensional clean predictor d maps noisy1536-dimensional target codes,
continuous time and the owning4096-dimensional contextual state to clean image
codes. Training uses uniformly sampled interpolation time and the clipped
velocity-equivalent clean-code error, plus balanced continuation/terminal loss.
These are established flow/denoising components, not a new learning-theory
claim. Model labels and history are image-derived continuous coordinates.

Teacher forcing can process all answer-image observations in parallel. Source
parameters stay frozen, but gradients must pass through them into the image
input adapter. Each contextual layer is checkpointed independently. A
functional prefix cache concatenates immutable prompt KV with that call's new
KV without mutating the prefix: backward recomputation cannot accidentally
append the same future twice. Explicit masks exclude future observations and
per-example prefix padding. No source vocabulary projection is instantiated.

At inference, check learned stopping before each block, sample clean codes,
normalize each tile, decode and threshold physical pixels, then encode the
actual emitted pixels. Only those reread image codes enter the learned image
encoder. Advance native KV by the two new tile observations, retaining all
prompt keys; do not recompute the entire image history. The unchanged uniform
safety cap is128 blocks. The output is a physical image stream.

Only image input/output/stop parameters are saved in adapter checkpoints;
frozen source and codec weights are reused by authenticated reference. Full
prompt KV is substantially larger than a final-layer suffix memory. A complete
corpus KV cache must not be written without measuring its size and available
storage. Online or bounded replay caches can support later training without
duplicating source weights. Efficient operation is a measured requirement,
not a property established by this design.

Before training, qualify the actual inherited model's numeric forward path,
checkpointed input gradients, prefix immutability, padding and future masks,
parallel/incremental agreement, head checkpoint round-trip, physical output
feedback and structural stop. Untrained execution receives no language credit.
The actual end state still requires trained complete answers on independent
prompts, useful language behavior, faithful writing including forms outside
the training inventory, and measured single-GPU feasibility.
