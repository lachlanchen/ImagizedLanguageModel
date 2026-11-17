# Nonlinear feedback calibration improves error slightly but fails its gate

The complete head-only comparison is terminal. Both arms complete twenty epochs
and 2,660 actual AdamW updates on all 33,941 sequence/slot rows. Weighted native
feedback error falls from **.639791851** initially to **.578608694** for the
affine control and **.542496895** for the nonlinear residual. The .036111799
gap fails the prescribed .10 improvement; final error also exceeds .10. Every
complete-record condition fails. This does not qualify a reader or language fit.

| Complete training record | Native positions | Initial error | Final affine | Final nonlinear |
|---|---:|---:|---:|---:|
| 513-glyph book | 509 | .591157 | .532191 | .504267 |
| 846-glyph QA | 740 | .572978 | .500330 | .462209 |
| 1,042-glyph QA | 961 | .654803 | .587653 | .537138 |

These errors concern source embedding coordinates, not glyph readability,
semantic answers or language generalization. All data are from fitting sources.
No answer image, autonomous continuation or development/DRCD record is generated
or scored. Original language results and frozen readings are unchanged.

The [prepared complete support](evidence/nonlinear_feedback_preparation_20261118.json)
covers all 13,785 source records and weights from 8,172 original batches:
5,749,578 span appearances and 5,844,979 native positions reduce to 29,755
native sequences and 29,740 physical span inputs. All aliases remain separate
targets sharing image features. Each arm visits 678,820 derived rows across
twenty epochs; those visits are not unique language facts or full-model updates.

The actual frozen image encoder supplies three BF16 queries per physical span.
All initial affine/nonlinear predictions match exactly on cached rows. The first
eight singleton checks have relative query RMS .000775--.005210, within .01;
this finite check does not prove deployment equivalence for all spans/layouts.
All three complete-record two-update trials open the new branch, remain finite
and reset exactly. Original encoder and codec tensor hashes remain unchanged.

The [source-bound result](evidence/nonlinear_feedback_alignment_result_20261118.json)
preserves all forty epoch measurements, common permutation identities, final
heads, cache and preflight. It verifies each row is visited once per epoch,
paired orders match and actual optimizer counters reach 2,660. No intermediate
head is selected. Only the new coordinate heads are fit; the original visual
features, query attention, group/count predictors and source statistics are
fixed. No contextual body is loaded during this head-only stage.

CPU preparation takes 24.551 seconds at 675,688,448 bytes peak RSS. The single
GPU pipeline takes **57.422 seconds**, with **.685 GiB** peak allocated VRAM,
no pressure waiting and **54.359 GiB** minimum available host RAM. Its supervisor
scope is 70.075 seconds. These nested scopes are not additive. The heads have
789,248 and 10,754,816 trainables; eventual complete model footprints would be
1,465,381,142 and 1,475,346,710 parameters, including inherited pretraining.
Head calibration costs do not establish a small foundation or matched-quality
efficiency. The owner exits and GPU0 is released; foreign services are unchanged.

The result rejects this twenty-epoch calibration as sufficient support. Removing
an affine-family constraint does not guarantee that finite optimization reaches
useful alignment. The remaining loss could reflect feature limitations, output
fitting, precision or their interaction; these data do not identify a unique
cause. Do not extend unchanged epochs or relax the threshold.

The [next separate procedure](feedback_ridge_refit_protocol_20261118.md) keeps
both prescribed final feature maps and sources fixed and refits only their
output coefficients by weighted ridge regression. It tests a concrete alternative
to further stochastic epochs. Its exact-arithmetic solution and actual BF16
execution must be reported separately. No working ILM or further language fit
is admitted by this failed calibration.
