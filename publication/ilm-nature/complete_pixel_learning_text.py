"""Source-bound complete-answer result text for the staged manuscript."""

SUMMARY = (
    'Written language includes handwritten and historical forms that cannot always be specified by a character code. '
    'We investigate an independent model that reads writing images and generates answer images, using encoded text only for offline preparation or separate diagnostics. '
    'Controlled experiments distinguish faithful reconstruction, answer selection and autonomous writing on one consumer GPU. '
    'A frozen continuous codec reconstructs 126 of 128 development blocks exactly, while earlier complete image writers fail useful answering. '
    'We couple full native visual context and emitted-image history to a normalized causal density over physical pixel patterns. '
    'The externally pretrained 7.544B actor adapts 13.856M parameters and reaches 32/32 pixel-exact complete training answers after 4,000 updates, at 18.191 GiB peak allocation. '
    'Fresh-PNG evaluation gives no exact answers on 16 Chinese development questions or eight English transfer instructions, and the outputs retain broken glyphs. '
    'A separate cold deployment reproduces five complete streams exactly without a runtime teacher, OCR or answer font renderer. '
    'Complete emission for learned conditions is achievable, but useful language generalization remains unresolved. '
    'Historical-script understanding, general instruction following and efficiency at matched language quality remain unachieved.'
)

RESULTS = r'''The native actor learns complete answer images
(Supplementary Note 57; Figs.~S34 and S35). Full prompt context conditions a
causal pixel density and learned stop; emitted images become history.
The externally pretrained 7.544B actor adapts 13.856M parameters on 32 natural
Chinese questions from 16 passages. Exact complete training answers increase
from 0/32 to 17/32, 29/32 and 32/32 at updates 0, 1,000, 2,000 and 4,000.
The registered exact-fit stop triggers at 4,000; fresh-PNG evaluation confirms
all 32 answers and all 16 passage pairs. However, reused development gives
0/16 exact Chinese answers and 0/8 English transfer answers, with foreground
F1 0.240 and 0.034. Development outputs contain broken glyphs and unrelated
training fragments. All outputs stop below the
safety cap. Active optimization takes 63.2\,min and total execution 81.6\,min,
at 18.191\,GiB peak allocated memory, excluding inherited pretraining.
Five cold-loaded PNG replays preserve complete pixels and stopping exactly.
Complete-image fitting and deployment pass; new-prompt answering fails.
Changes in initialization, decoder, precision and schedules preclude attributing
this gain to the likelihood alone.
'''

DISCUSSION = r'''The subsequent complete pixel actor does reproduce all 32 seen answers
from their original prompt images. Its failure on all 24 normal development
cases therefore separates exact training emission from useful conditional
generalization. More unchanged fitting of those 32 questions is not supported
as the next learning step. Varied visual-language supervision must be tested
against readable, correct answers to new prompts, while preserving the
independent image generation path and accounting for external pretraining.
'''

NOTE = r'''\clearpage
\section*{Supplementary Note 57 | Complete native pixel-language learning}
\label{note:complete-pixel}
The complete actor retains the native source's full image-prompt key/value
context. Scoped suffix adapters operate only after prefill; original prompt
states remain unchanged. A learned start query and image-history projection
condition one 4,096-dimensional causal state. A normalized physical density
replaces the earlier image-code predictor. The frozen image codec encodes
observed history only; its image decoder and previous prediction flow are
disabled in output generation. The actor contains 7,543,984,691 parameters,
of which 13,855,681 train, including 2,666,944 in the physical density and
6,193,152 suffix-adapter parameters. External foundation pretraining and prior
image-interface/raster learning remain part of initialization.

Let $X$ denote the observed prompt image, $Y_t$ a binary $32\times64$ answer
block and $h_t=R(X,Y_{<t})$ the causal contextual state. Lossless packing
maps each block to 256 adjacent $2\times4$ pieces, $b_{tj}\in\{0,\ldots,255\}$.
These are eight-bit physical patterns, not character or word identities.
The four-layer width-192 causal head gives
\[
 p_\theta(Y_t\mid h_t)=\prod_{j=1}^{256}
 p_\theta(b_{tj}\mid h_t,b_{t,<j}).
\]
Each conditional is normalized over all 256 patterns. Exact inverse packing
contains no font or glyph inventory. The state receives per-state layer
normalization and a learned 4,096-to-192 projection. Small-head arithmetic
uses FP64 with FP32 stored parameters after a separately qualified numerical
repair; the large native body retains its source precision. Cached/full
piece logits agree within $1.75\times10^{-13}$ in that qualification. This
numerical check is not evidence for a language-learning mechanism.

With $s_t=\sigma(w^\top h_t+c)$, a finite stopped answer has probability
\[
 s_{T+1}\prod_{t=1}^{T}(1-s_t)\,p_\theta(Y_t\mid h_t).
\]
Normalization over finite answers additionally requires almost-sure stopping;
this study does not prove that condition. Deployment uses threshold 0.5 and
a uniform 128-block cap, recording a cap event as failure. No desired answer
length enters generation. Every next state reads the actual emitted image,
not the desired target, source character ID or predicted latent image code.

For question $i$ with $T_i$ target blocks and pre-decision stop logit $a_{it}$,
the implemented objective is
\[
\begin{split}
\mathcal L={1\over B}\sum_{i=1}^{B}\bigg[&
 -{1\over256T_i}\sum_{t=1}^{T_i}\sum_{j=1}^{256}
 \log p_\theta(b_{itj}\mid h_{it},b_{it,<j})\\
 &+{1\over2T_i}\sum_{t=1}^{T_i}\operatorname{softplus}(a_{it})
 +{1\over2}\operatorname{softplus}(-a_{i,T_i+1})\bigg].
\end{split}
\]
Question/block means and balanced continue/end terms are explicit weights,
not the unscaled joint finite-answer NLL. Training supplies correct preceding
images and shifted physical pieces. All complete target pieces are scored.
Owning native coordinates and padding masks preserve variable-length
causality. Tested future-image and future-code padding exclusion are exact.

The fixed cohort reuses 32 natural Chinese CMRC questions, with two distinct
questions per each of 16 passages and 107 complete answer blocks. Development
contains the first eight prepared CMRC article pairs (16 questions) and one
English Dolly question from each of eight categories, fixed before learning.
All reference alternatives are retained. These are reused development data,
not sealed tests; English receives no training in this stage. Chinese controls
erase question ink and English controls erase prompt ink while keeping normal
page geometry. No development targets or case identities enter generation or
optimization.

Batch four uses AdamW $(.9,.95)$, decay .01, gradient clipping at one and
fresh full-cohort permutations. Physical-head and other peak rates are
$3\times10^{-4}$ and $10^{-4}$, with 100-update warmup and a cosine schedule
declared over at most 8,000 updates, ending at one tenth of peak. The only
early-stop rule is all 32 complete training answers exact at registered
updates 1,000/2,000/4,000/8,000. It triggers at 4,000, independently of
development quality. This retains 16,000 draws and 500 exposures per question.
Actual update-two model/Adam/sampler/loss/gradient restore-replay is bit-exact,
adding one executed call and four examples: 4,001 calls and 16,004 examples.

Matched whole-cohort correct-history image NLL is 9.883952, 0.006959,
0.001138 and $8.659788\times10^{-9}$ at updates 0/1,000/2,000/4,000.
Autonomous exact training answers are respectively 0/32, 17/32, 29/32 and
32/32, with F1 .130254/.943528/.996086/1. All 32 original prompt-prefill
states are bit-exact after learning. Intermediate generation uses that
authenticated replay pool; final training and all development generation
invoke the complete PNG reader again.

\begin{center}\begin{tabular}{lrrrr}
Final complete answers & Cases & Exact & F1 & Stops\\\hline
Training, fresh PNG & 32 & 32 & 1.000000 & 32\\
Chinese development & 16 & 0 & .239785 & 16\\
Chinese, question erased & 16 & 0 & .172495 & 16\\
English transfer & 8 & 0 & .034482 & 8\\
English, prompt erased & 8 & 0 & .048394 & 8\\
\end{tabular}\end{center}
Both answers match in 16/16 training pairs and 0/8 development pairs. There
are no cap failures. Chinese normal-minus-erased F1 is .067290 and English
is $-.013913$. These physical differences do not certify semantic question
use. Original development outputs contain broken glyphs and some unrelated
training-answer fragments; all 80 final training/normal/erased streams and
their owning references are actually reviewed without repair.

Active optimizer time is 3,789.292\,s and full execution is 4,893.048\,s at
18.191\,GiB peak allocated memory on one RTX 4090 D. These exclude inherited
pretraining and prior raster learning. The independent CPU audit authenticates
all sources, complete images, stop events, permutation/exposure ledgers and
208 output streams across all stages. It re-encodes 668 actual feedback-image
records, with maximum CPU/GPU feature error $6.914139\times10^{-6}$ under
the registered $10^{-4}$ tolerance. It does not independently replay every
inherited neural forward or optimizer update, or assign semantic accuracy.

A separate cold deployment reconstructs the actor from its own final state,
authenticated frozen body, image-history encoder and numeric controls. The
first four original inputs plus the first longest training answer are fixed
without quality selection (indices 0/1/2/3/28). All five fresh-PNG replays
match complete pixels, stop probabilities, feedback hashes and reader hashes
exactly. Expected outputs are read only after generation is saved. This path
requires no legacy training/raster checkpoint, runtime teacher, OCR,
tokenizer or answer font renderer. It peaks at 17.529\,GiB and takes
96.675\,s including 76.547\,s loading. Deployment replay verifies restoration,
not language generalization. No historical-script, arbitrary writing-style,
lightweight-foundation or matched-quality efficiency claim follows.

\clearpage
\plate{Figure_S34_complete_pixel_learning}{\textbf{Complete image fitting and failed transfer.}
\textbf{a}, Native prompt context conditions a physical-pixel density, learned
stop and actual emitted-image feedback. Blue modules contain trained
interfaces; the native body and history encoder remain frozen.
\textbf{b}, Matched whole-cohort image NLL under correct preceding images,
on a logarithmic axis. \textbf{c}, Autonomous complete-answer counts separate
training reproduction from failed reused development. English is untrained
transfer. Pixel agreement does not assign semantic accuracy.}{fig:complete-pixel-learning}

\clearpage
\plate{Figure_S35_complete_pixel_examples}{\textbf{Fixed source-order training and development examples.}
The first two training and first two Chinese development questions receive
their complete original pages. Display excerpts contain the last 64 physical
input rows, including final question lines; crop boxes and full original
inputs remain in the evidence archive. Complete output/reference strips use
the same display scale and retain original pixel values. Training answers
match exactly, while unseen answers contain broken glyphs. No output quality
selects these examples. All erased controls and every failure are retained.}{fig:complete-pixel-examples}

Exact preparation, training, audit and inference commands are in
\path{references/complete_pixel_language_protocol.md},
\path{references/complete_pixel_language_result.md} and
\path{references/complete_pixel_language_inference.md}.
Original images and audit receipts are in
\path{references/evidence/complete_pixel_language_result/} and
\path{references/evidence/complete_pixel_runtime_qualification/}.
'''
