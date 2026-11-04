# Reading images and using their language

Primary-source check, 2026-11-05. These papers inform the interpretation of the
current experiment; they do not change V57's frozen protocol or thresholds.

[Shi, *VLMs Read Aloud*, arXiv:2611.01517v2, 3 November 2026](https://arxiv.org/abs/2611.01517v2)
reports a gap between transcribing rendered context and answering questions
from it. The reported recognition and use thresholds differ even within the
same VLM; its proposed remedy uses a textual transcript. This preprint supports
testing use separately from recognition. It does not establish that the gap
must persist in an independently trained image-only model. V57 tests natural
answers and question interventions directly; it does not use the transcript
remedy at inference. The claims here are based on the primary abstract, not
an independent reproduction of its benchmarks.

[Zhang, de Lhoneux and Poelman, *Seeing the Unseen*, arXiv:2608.30541v1](https://arxiv.org/abs/2608.30541v1)
studies pixel-model adaptation to Tibetan. It reports that orthographic
proximity and the starting pretraining model affect semantic transfer.
Its implication for ILM is a hypothesis: arbitrary representability alone
does not grant semantic competence in an unfamiliar script. A canonical
Chinese result must not be promoted to oracle, seal, handwriting or Khitan
competence without transfer measurements. No adaptation experiment from this
paper has been reproduced here.

[Sun et al., *LargePiG*, arXiv:2410.11366v1](https://arxiv.org/abs/2410.11366v1)
uses existing LLM attention and vocabulary distributions for a training-free
pointer-generator, reporting improvements in query and document tasks. It is
relevant prior work on content selection and copying, but its vocabulary-based
runtime differs from V57's image-equality copy distribution. A source pointer
is an established mechanism, not a novelty claim for this repository.

[Zhang et al., *PixelBench*, arXiv:2611.02540v1, 2 November 2026](https://arxiv.org/abs/2611.02540v1)
evaluates explicit raster generation through palette and grid indices emitted
by an LLM's existing vocabulary. It also reports pixel-reconstruction
post-training of Molmo2-4B. This is relevant auxiliary-supervision evidence;
it does not meet ILM's image-only deployed interface by itself. We have not
downloaded its weights or changed the present experiment around its results.

The useful common distinction is between faithful marks, correct evidence
selection and meaningful sequence generation. A positive result in one
category does not automatically establish the others. V57 remains a bounded
grounded-reading test with a free raster branch, not proof of a general
replacement for a large language model.
