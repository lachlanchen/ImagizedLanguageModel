import Std

namespace CandidateCompetition

/-- The common log normalizer cancels when comparing the same two independent
candidates. A different candidate set can only change their order through
a change of their own relative scores (with a fixed tie convention). -/
theorem unchanged_order (u v normalizer₁ normalizer₂ : Int) :
    (u - normalizer₁ ≤ v - normalizer₁) ↔
    (u - normalizer₂ ≤ v - normalizer₂) := by omega

/-- Exact criterion for a previously losing target to defeat this rival after
adding context residuals. The initial gap and the two learned corrections
are observable before the common softmax. -/
theorem overturn (u v δu δv : Int) :
    v + δv < u + δu ↔ v - u < δu - δv := by omega

/-- When a target beats every rival by more than twice a displacement bound,
no context intervention within that bound can reverse its win. -/
theorem certified_winner {C : Type} (s d : C → Int) (target : C) (b : Int)
    (lower : -b ≤ d target) (upper : ∀ c, d c ≤ b)
    (gap : ∀ c, c ≠ target → 2*b < s target - s c) :
    ∀ c, c ≠ target → s c + d c < s target + d target := by
  intro c hc
  have h := gap c hc
  have u := upper c
  omega

end CandidateCompetition
