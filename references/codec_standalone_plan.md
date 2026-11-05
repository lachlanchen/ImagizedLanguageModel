# Independent loading of the complete codec-language actor

Keep the live V68 trainer and autonomous-audit sources unchanged. The current
completed-state loader authenticates a shared GGUF archive at startup. A final
deployment must instead load its own native weights and source modules without
that archive, GGUF reader, tokenizer, renderer, data cache or model service.

Reconstruct the exact native architecture on the PyTorch meta device, using
the authenticated vision/text configurations, FP32 visual weights, BF16
contextual storage, stable contextual operators and the same codec/adapters.
Match all 7,539,239,282 parameters, the 9,351,937 trainable subset and all state
names/shapes/dtypes. Nonpersistent rotary buffers must be explicitly rebuilt
after loading; they cannot remain meta tensors.

Export from the existing final actor after its autonomous audit and independent
verification, requiring the original bounded Chinese development gates to pass.
Keep inherited pretraining and remaining general/historical-language limits
explicit. Do not copy full weights while training runs. On an idle GPU 0, load
the accepted actor once and stream its exact state into bounded native tensor
shards, with source, tensor and file hashes. Retain its original codec encoder
for architecture identity; autonomous generation never calls that label path.

Copy only the necessary runtime modules into a new bundle. Bundle loading
uses no repository paths or shared initialization archive. It validates the
entire state inventory and each tensor's physical bytes before assignment.
Keep the installed Python/library requirements explicit, without duplicating
an existing environment. A separate fresh-process verifier must replay the
first fixed autonomous normal-case output exactly, including raw blocks and
learned stop events, from an empty working directory and empty model caches.
Its audit hook blocks network/subprocess calls and access to source/data roots,
allowing only the bundle and explicit prompt/output paths within those roots.

Before final weights exist, run only a CPU structural/shard preflight. Validate
the complete real architecture against authenticated native metadata, and
round-trip the actual frozen codec's parameters and supplied writing codes in
a temporary, isolated bundle. Include corrupted/missing/duplicate state
rejections. This does not execute the complete contextual model, establish
final-actor isolation or qualify language. No second heavy model job is
permitted while the trainer owns GPU 0.
