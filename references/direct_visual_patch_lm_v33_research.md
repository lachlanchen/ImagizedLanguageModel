# Direct Visual Patch Language Model V33: Research Decision

Date: 2026-08-13

Status: decision recorded before V33 implementation or measurement

## Decision

V33 will replace V32's learned glyph bottleneck with direct autoregressive
prediction of binary writing patches:

```text
prompt raster -> 32 x 32 patches -> causal transformer -> next 32 x 32 patch
                                                      -> generated raster strip
```

The deployed student accepts pixels and patch-presence masks and returns pixels
and a scalar stop decision. It has no tokenizer, character IDs, Unicode IDs,
OCR, vocabulary embedding, vocabulary output head, glyph lookup, visual
codebook, VAE, or external language model in its inference path.

This is intentionally close to the strongest demonstrated pixel-language
baseline instead of another novel bottleneck. Existing work that works is an
asset when its role, license, and runtime boundary are explicit.

## Why V32 Must Not Be Scaled Yet

The exploratory V32 run completed 4,500 finite BF16 updates in 195.08 seconds
and used 1.21 GiB peak allocated VRAM. Autonomous development accuracy was
effectively zero. More importantly, an evaluator-only target-reconstruction
test gave only 13.53% character accuracy even when the target answer raster was
provided directly to the target encoder.

That diagnostic isolates the first failure before semantics:

- two stride-2 convolutions reduced each 24 x 24 glyph to 6 x 6 features;
- global average pooling then discarded that spatial arrangement;
- a moving, context-conditioned 32-dimensional state had to preserve the
  remaining glyph identity;
- a jointly moving decoder reconstructed the raster from that state; and
- a diagonal Gaussian planner averaged plausible states during generation.

A longer V32 run would not be a clean language experiment. V33 removes the
target encoder, latent state distribution, and raster decoder as separate
failure surfaces. Its hidden state maps directly to 1,024 binary pixel logits.

## Prior Art Audit

### PIXAR

[PIXAR](https://arxiv.org/abs/2401.03321) established decoder-only,
pixel-input/pixel-output autoregressive language modeling. Its public
checkpoint has 12 LLaMA-style layers, width 768, 12 attention heads, and a
3,072-dimensional SwiGLU intermediate layer. Input and output projections map
8 x 16 monochrome patches to and from the hidden stream. The released archive
contains 113,471,488 parameters excluding any vocabulary embedding.

The audited source repository is
[`april-tools/pixar`](https://github.com/april-tools/pixar) at commit
`810a423336d5fdeb33e4c1695381e357ff32c4bb`. Its current source license is MIT.
The separately hosted Google Drive checkpoint does not carry an explicit model
weight license in the archive. V33 may use it for local research initialization
but must not redistribute it or imply that the source-code license governs the
weights.

Checkpoint receipt:

| Item | SHA-256 |
|---|---|
| downloaded archive | `509a01f43b2e6eff1ff03dde99f056a15fc758ed2829e6f3155980ce9b97c7d3` |
| `backbone/pytorch_model.bin` | `ae4f899bbbb0bfaa90ee033c6d1dc5aeb3f50b323f726a0df241be8104682eb9` |
| `backbone/config.json` | `1bc49b2e8cc59a4865d0f737739e73a55d14a6b849a5cae516de3dbb3f8e2ace` |

An exact modern `transformers` port loads all 109 transformer tensors and the
final RMSNorm. The only omitted tensors are PIXAR's 8 x 16 input and output
projections, which require an explicit resolution adaptation.

### MIXAR

[MIXAR](https://arxiv.org/abs/2604.11575) is the most relevant direct
successor. It reports that 8 x 8 patches lose CJK stroke detail and uses
32 x 32 binary patches with font size 32. Its 116M configuration retains 12
transformer layers; its larger configuration has 477M parameters and 24
layers. Both use direct linear pixel output, not a VAE or discrete image
tokenizer.

MIXAR reports multilingual pretraining on 138B patches for one million updates,
followed by a short context-aware adversarial stage. The paper promises code,
data, and checkpoints after acceptance, but no public release was discoverable
on 2026-08-13. V33 therefore reproduces the disclosed mechanism without
claiming or depending on unavailable MIXAR weights.

### Continuous image generation

[MAR](https://arxiv.org/abs/2406.11838) shows that autoregression does not
require vector quantization: a diffusion loss can model each continuous image
token. This is the preferred fallback if deterministic binary BCE produces
averaged or noisy writing after the causal mechanism passes. It is not included
in the first V33 arm, because adding a diffusion network before verifying the
direct-patch baseline would make failure harder to localize.

## Model

Let a rendered monochrome strip be divided into non-overlapping patches

\[
x_i\in\{0,1\}^{32\times32},\qquad i=1,\ldots,L.
\]

White is one and ink is zero, matching PIXAR's binary convention. A convolution
whose kernel and stride are both 32 maps each raw patch to the model width:

\[
e_i=W_{in}\operatorname{vec}(2x_i-1)\in\mathbb R^{768}.
\]

A 12-layer causal transformer computes

\[
h_i=F_\theta(e_{\leq i}),
\]

and a direct linear image head predicts the next patch:

\[
\hat x_{i+1}=\sigma(W_{out}h_i)\in[0,1]^{1024}.
\]

A one-dimensional head predicts whether the current patch ends the response.
This is a control signal, not a textual EOS token. During generation, every
predicted raster patch is appended to the same visual stream and reread by the
model.

The 8 x 16 PIXAR projections are resized to 32 x 32 only as initialization.
The transformer core is loaded exactly. A random-core control separates
architecture from external pretraining.

## Rendering And Causal Boundary

The first proof uses a 32-pixel-high strip. Offline PIL rendering uses Noto CJK
fonts at 24--28 pixels and binary antialiasing. Prompt and answer occupy
separate patch regions. Their horizontal origins are jittered inside patches,
so a patch is a retinal field and not a character slot.

For supervised records, the offline data builder renders:

```text
[question pixels][visual answer marker][answer pixels]
```

The student batch contains only the resulting raster, attention mask, answer
loss mask, and stop targets. Text strings remain in evaluator metadata and
never enter a model method. Natural continuation records are rendered without
question/answer labels.

## Staged Proof

### Stage A: visual interface calibration

Train the resized input/output projections to reconstruct observed patches at
the same positions while the PIXAR transformer core is frozen. This is a
calibration objective, not evidence of language. It must pass on held-out fonts
and unseen text before causal training starts.

### Stage B: Chinese causal continuation

Train next-patch prediction on the public-domain Chinese Wikisource corpus.
The target is shifted by exactly one visual patch. No character boundary or
character label is exposed.

### Stage C: visual instruction continuation

Train answer-only next-patch loss on short Chinese Alpaca records. The existing
source is CC BY-NC 4.0, so this checkpoint is a non-commercial research
artifact. Future evidence must replace it with public-domain or permissively
licensed teacher-audited book questions.

### Stage D: readability repair, only if indicated

If prompt conditioning passes but generated patches are noisy, add either the
PIXAR context-aware adversarial objective or a MAR-style per-patch diffusion
head. This stage requires a separate protocol amendment and may not be used to
reinterpret a failed V33 result after inspection.

## What V33 Can And Cannot Establish

A positive bounded result establishes that one compact, independently runnable
student can read a Chinese prompt raster and generate a prompt-dependent answer
raster without symbolic units at inference. It does not establish parity with
Qwen, broad world knowledge, superior training efficiency, page-scale reading,
or historical-script understanding.

A negative result remains useful:

- calibration fails: the resized visual interface is inadequate;
- calibration passes but continuation fails: causal transfer is inadequate;
- continuation passes but instruction prompts fail: the dataset or semantic
  adaptation is inadequate;
- teacher-forced patches pass but autonomous output fails: exposure bias or
  image-density modeling remains inadequate.

