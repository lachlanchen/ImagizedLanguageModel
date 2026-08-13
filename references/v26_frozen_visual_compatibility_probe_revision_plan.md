# V26 Frozen Visual Compatibility Probe: Revision Plan

Date: 2026-08-13

Status: completed post-result diagnostic revision

## Scope

Publish a reproducible diagnostic that distinguishes two explanations for the
rejected V26 result:

1. the V26 image-derived context state contains next-glyph information that its
   stochastic particle proposal suppresses; or
2. the state changes with history but does not contain a transferable relation
   to the associated next-glyph image.

This diagnostic is not preregistered evidence, cannot select a V26 checkpoint,
and cannot authorize access to the frozen split or training of a writer.

## Changes

- Add an image-only candidate-compatibility probe with separate appearance,
  history-residual, and fused-state controls.
- Freeze every V26 parameter and train only the 1.11M-parameter probe for one
  pass over the existing 16,384 train suffix pairs.
- Evaluate on the fixed 512 development suffix pairs using two cross-font
  assignments per pair.
- Give exact score ties half credit and report strict accuracy and tie rate.
- Add a raw-retina cross-font identity control on the same candidate pairs.
- Record source, checkpoint, split, pair, font, runtime, and boundary receipts.
- Publish the result in the README, goal document, active paper, and a
  deterministic evidence-derived figure.

## Interpretation Rule

An exploratory arm accuracy above `0.65` for frozen history or fused state would
support preserving the V26 encoder while replacing its stochastic proposal.
A chance result with retinal identity at or above `0.99` instead supports joint
retraining of context representation and deterministic candidate compatibility.

No outcome establishes language capability, compute efficiency, or generation.
