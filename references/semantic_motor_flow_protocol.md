# Joint continuous glyph motor: prospective flow experiment

Preserve the date-capable complete-glyph checkpoint. Both broadened readout
compilers and the noise-trained deterministic decoder failed their gates.
Repeated E/D passes do not improve the five normal reference first forms above
0.95F1. These results motivate testing a *joint conditional distribution* over
whole motor codes instead of another deterministic point decoder. They do not
prove that this particular flow will solve the failure.

## Model and theory

Let a be the RMS-one direction of an existing training semantic anchor, c its
corrupted continuous condition, and epsilon an independent standard Gaussian
in R^4096. For sigma_min=0.001 and t uniform in[0,1], use

    x_t = [1-(1-sigma_min)t] epsilon + t a
    u_t = a - (1-sigma_min) epsilon
    L = E || v_theta(x_t,t,c) - u_t ||^2 /4096.

The endpoint is a narrow Gaussian around a. Under the usual conditional-flow
matching assumptions, the optimal velocity is the conditional expectation of
u_t at each path state, and its continuity equation transports the noise law
toward the conditional endpoint law. This predicts a *distribution of complete
vectors*, rather than an average final glyph image. Finite neural training,
ODE integration and a discrete empirical dataset do not supply an exactness
or semantic-correctness guarantee. See the primary
[Flow Matching paper](https://arxiv.org/abs/2210.02747) and
[Rectified Flow paper](https://arxiv.org/abs/2209.03003).

A new dense velocity model has width4096, three residual blocks, sinusoidal
time encoding and a full4096-coordinate output. Each residual uses LayerNorm,
two4096-square linears and SiLU; its contribution is scaled0.1. Noisy and
conditional inputs start with identity linear maps; the final output starts
at zero. Width equals state dimension to remove the fixed narrow output-span
ceiling encountered by earlier continuous writers. This is representational
capacity, not proof that a finite fit learns the desired transport.

The language body, G,E,D,affine buffers and stop predictor are unchanged. The
motor has no vocabulary projection, indexed glyph table or font. Its condition
is a continuous semantic vector. At deployment it samples four motor vectors
with32Heun intervals (Euler on the last interval), decodes each through the
preserved neural D, and re-encodes each *actual binary image* with E. Select
the candidate with greatest cosine to the original condition. This selection
uses only four neural samples and learned image features, never a character
bank or reference. Actual selected pixels supply all subsequent feedback.

## Training

Only the existing4,087 training-only anchor/image forms supply data; no QA
question, answer, development hidden state or source-output teacher is loaded.
For60% of draws, mix the RMS-one target with a uniformly different form, with
distractor weight uniform[0.25,0.45], plus0.05Gaussian noise. For20%, add
Gaussian noise with sigma uniform[0,0.8]; the remaining20% have a clean
condition. Normalize the resulting condition to RMS one. This creates motor
uncertainty, not new language examples or unseen glyph identities.

Seed202611107. Train8,000batch128 AdamW updates, learning rate1e-4,
betas(0.9,0.99), weight decay0.01, clip norm1,200-step warmup and cosine decay
to10%. Parameters and loss are FP32; linear matmuls use BF16 autocast. Use
unweighted velocity MSE, without target-dependent ink weighting or endpoint
reconstruction loss, to preserve the stated flow-matching objective. EMA decay
is min(0.999,(1+step)/(10+step)); only the final EMA is sampled. Record exact
parameter count, full costs and the base checkpoint hash. No efficiency-parity
or from-scratch language claim is permitted.

## Qualification before language inference

Assess all4,087 anchors under clean conditions and a fixed0.55/0.45 mixture
with distractor offset1049. Seed202611108,32steps,four candidates. Keep all
candidates, their continuous image-encoder scores, selected indices and all
selected pixels. The canonical target is evaluator-only. Report selected
physical foreground F1/exactness. Also report the best-reference candidate F1
as an explicitly *oracle diagnostic*; it must never select deployed output or
be reported as actual accuracy.

Require clean mean F1>=0.99 with95%of forms>=0.95F1, and mixture meanF1>=0.95
with80%of forms>=0.95F1. Inspect fixed64-triplet boards. A pass qualifies only
this known-form motor, not language. A subsequent separately documented
forty-stream assessment must use the original Chinese prompts/erased controls,
autonomous learned stop, unchanged64-glyph cap and actual sampled-image
feedback. Until then the original language checkpoint remains the verified
limited working model. No result is claimed in advance.

Use the existing sole GPU0 mutex; no concurrent project model jobs. This
separate motor stage requires48GiB host entry/32GiB runtime reserves,8GiB
allocation cap,1,800seconds per stage, <=0.5/0.1some/full PSI and at most300s
pressure waiting. Inspect foreign ownership first. No original training
admission or completed source file changes. Preserve failed/partial runs and
stop exact project runtimes after completion.
