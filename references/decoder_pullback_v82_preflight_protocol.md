# Preflight a state-only physical writing metric

Prospective derivative and runtime check, 7 November 2026, following the
failed matched readout repair. No model optimization, new generation result,
or quality improvement is claimed by this preflight.

The implementation computes per-block

    L(v;x) = mean(v²) + 0.001 mean((J_R(x)v)²),

where v is predicted-clean minus target code and R is the frozen writing
decoder including the sampler's actual per-tile layer normalization. The
metric point x is the observed noisy code, detached from optimization; it
does not depend on the current predicted endpoint. The positive code-MSE
term preserves positive definiteness. Given x,t,h, the function-space
minimizer remains the conditional target mean. This does not imply faster
finite-model learning or language capability. No target-dependent ink weights
or pixel BCE are introduced.

Use every block of the first four questions in the existing fixed cohort,
positions 00–03, from the completed V81 adapted full-history trace. This
gives ten blocks in the first two question pairs, without selection by error
or quality. Authenticate the completed evaluation report, CPU verification,
source headers and original code records. At each t in [0,0.5,0.9], construct
x=(1-t)epsilon+t*z using CPU seed 20268200 and the same noise for every time.
Use the retained predicted-code minus target-code residual as the derivative
direction; no network output is changed.

Before any training, check the actual frozen codec:

- strength zero reproduces code MSE and its gradient exactly;
- the computed residual gradient equals the independent reverse-mode
  expression 2v/1536 + 2*0.001 J_R(x)^T J_R(x)v/2048, within relative L2
  error 2e-4 and absolute max 1e-4;
- FP32 forward-mode JVP agrees with FP64 central differences through a
  separately copied decoder at step sizes 1e-3 and 5e-4, within relative
  L2 error 2e-4 for both steps;
- the observed noisy input has no gradient through the metric, codec weights
  remain frozen and bit-identical, and every loss/gradient is finite;
- direct decoder access in FP32 reproduces the physical codec's logits
  exactly after the same normalization.

Report actual components, gradient norms, all discrepancies and wall time.
Record process memory and, for a separate GPU0 invocation, allocated VRAM.
These are small decoder-only costs, not complete-actor throughput. Preserve
failed checks and use a fresh output directory for any explicit revision.
Commit the implementation and protocol before executing. CPU first; only
if it passes may a single GPU0 decoder check run. No other ILM model job may
overlap. A derivative preflight pass permits further investigation, not a
language-success claim or an automatic full-corpus training run.
