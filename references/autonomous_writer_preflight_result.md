# Full native sampling gradients fit on one 4090; writing remains unqualified

The completed two-update execution check differentiates a final-image loss
through all 50 native sampling steps on GPU0. Peak allocation is **12.032932
GiB**, and the invocation takes **126.159 seconds**, including native replay,
two real optimizer updates and before/after images. This verifies an executable
learning path. It does not establish reliable writing or a useful ILM.

The [authenticated export](evidence/autonomous_writer_preflight_result/result.json)
records hashes of all 100 checked trajectory tensors (50 Chinese and 50
English), both native comparisons, the original full report and update journal,
and four actual unrepaired output PNGs.

## Native equivalence and the input-precision correction

Before either optimizer update, the custom sampler reproduces every native
step tensor and final pixel bit-exactly for the first Chinese and English
inputs. Both native outputs also equal the previously retained checkpoint
outputs. The sampler receives only native input-image features, a constant
generic condition and independent noise, with guidance one and a 512x256
answer canvas. No answer target, target code or label enters sampling.

The actual first autograd/checkpointed training forward also reproduces the
native final pixels bit-exactly before its optimizer update. The later English
training forward uses parameters already changed by the Chinese update and
is not presented as a replay of the initial checkpoint.

The first preflight, at implementation `d388371`, fails before optimization.
The [correction record](autonomous_sampler_correction.md) preserves the exact
threshold. Historical input features were encoded outside autocast, while
native inference encodes under BF16 autocast. For both checked examples their
maximum difference is .0625. Fresh encoding in the native precision context
restores exact sampler equivalence. This numerical correction is not a causal
diagnosis of the earlier content errors.

## Real final-image optimization

The [registered objective](autonomous_writing_protocol.md) is training-reader
CE on native display pixels plus ten times a balanced ink/background L1 loss
on the unclipped decoded RGB. The latter keeps a gradient on saturated white
and penalizes visible extra/missing writing that the reader can ignore. The
reference page and string enter only the post-generation losses. The canonical
pixel target limits this experiment's style claim.

| Observed execution quantity | Chinese update | English update |
| --- | ---: | ---: |
| Recognition CE | .021000 | .022402 |
| Balanced pixel L1 | .101051 | .113958 |
| Total loss | 1.031506 | 1.161979 |
| Step-0 prediction gradient norm | .002193 | .001550 |
| Step-24 prediction gradient norm | .002696 | .002887 |
| Step-49 prediction gradient norm | .013549 | .011441 |
| Input-reference gradient norm | .223564 | .214080 |
| Adapter gradient norm before clipping | 6.036190 | 10.239990 |

All inspected gradients are finite and nonzero. Both real AdamW updates use
learning rate 1e-5, decay .01 and global gradient clipping one. All 206 adapter
tensors change. Source transformer, VAE and reader parameters remain frozen
and without gradients. Full-trajectory autograd is used; this is not a detached
last-step approximation. The deterministic Euler update is reproduced outside
the checkpointed neural field so scheduler state cannot advance again during
backward recomputation.

The [Chinese before](evidence/autonomous_writer_preflight_result/before_zh.png)
and [after](evidence/autonomous_writer_preflight_result/after_zh.png) images
still contain incorrect writing. The [English before](evidence/autonomous_writer_preflight_result/before_en.png)
and [after](evidence/autonomous_writer_preflight_result/after_en.png) preserve a
readable phrase. These are execution examples from two seen inputs, not a
copying benchmark or evidence of semantic understanding.

The deployed transformer remains **3,899,903,488 parameters**, including
24,358,912 trainable adapter parameters, plus the 84,046,115-parameter VAE.
The 560,528,640-parameter frozen reader is training-only. External pretraining
is retained; this is not a small model trained from scratch or an efficiency
comparison at matched language quality. The method applies the established
[DRaFT principle](https://arxiv.org/abs/2309.17400) to this final writing loss.

The separately specified 32-update pilot starts again from the original
checkpoint, discarding these two preflight updates. Its full 64-case fresh
native inference and independent CPU evaluation are required before a learning
or writing conclusion. Natural instruction answering, broader scripts and the
general independent ILM objective remain unproved.
