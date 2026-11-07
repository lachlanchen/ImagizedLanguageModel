# Native-length readout of full-resolution window features

This post-window diagnostic is specified after the complete full-window
assessment: 0/32 Chinese and 2/32 English exact answers, despite readable
full-resolution codec reconstruction (0.086875 character error versus
0.705221 at 432). The native-432 source retains 11/32 English answers.
The observations leave a longer language-context interface among unresolved
limitations, in addition to visual-language alignment.

Keep the same full-resolution source pixels, codec samples, nine native
semantic windows and uniform overlap assembly. Before the unchanged language
body, adaptively average-pool the 64 by 64 fused visual feature grid in FP32
to 27 by 27 and cast back to the original dtype. Preserve the role/time
embeddings and rebuild the native complete-context mask for 729 visual
positions. The complete input still participates, but this feature resampling
is lossy and changes its spatial granularity. It is not equivalent to keeping
4096 independent image positions or proof of information preservation.

At native 27-square geometry, bypass pooling and require exact prefix, mask
and full decision replay on the same first four English normal pages as the
previous preflight. Assess all original 48 Chinese and 80 English conditions,
with unchanged seeds, instructions, native source weights, BF16 body, FP32
codec and 96-decision cap. No optimizer, new parameters, answer crops, OCR
conditioning, selective examples or teacher calls are allowed. Preserve all
outcomes; compare against both the full-window and native-resize conditions.

Any recovery would support this specific resampled source interface, not
demonstrate learned language or direct image answers. A failure rejects its
zero-shot use. This controlled feature-pooling idea is motivated by the
native-scale feature assembly in
[S2](https://arxiv.org/abs/2403.13043v2); it is not a replication of that
method or evidence of a new foundation architecture.
