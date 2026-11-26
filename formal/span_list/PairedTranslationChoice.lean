import Std

namespace PairedTranslationChoice

/-- Fixed-point stopped log scores, ordered by source row then candidate column. -/
def interaction (aa ab ba bb : Int) : Int := (aa-ab) + (bb-ba)

/-- Length-normalised row and candidate biases cannot explain this interaction. -/
theorem gauge_invariant (aa ab ba bb r s p q : Int) :
    interaction (aa+r+p) (ab+r+q) (ba+s+p) (bb+s+q) = interaction aa ab ba bb := by
  unfold interaction
  omega

/-- A single
shared offset on answer A can make *both* matched choices win by a prescribed
(non-strict) integer margin precisely when the interaction covers both margins. -/
theorem target_offset_exists (aa ab ba bb m : Int) :
    (∃ k : Int, m ≤ aa+k-ab ∧ m ≤ bb-(ba+k)) ↔ 2*m ≤ interaction aa ab ba bb := by
  unfold interaction
  constructor
  · rintro ⟨k,h₁,h₂⟩
    omega
  · intro h
    exact ⟨m-aa+ab, by omega, by omega⟩

/-- An interaction smaller than two margins cannot be repaired by any
combination of source-only and target-only offsets. -/
theorem needs_interaction (aa ab ba bb m : Int)
    (h : interaction aa ab ba bb < 2*m) (r s p q : Int) :
    ¬(m ≤ (aa+r+p)-(ab+r+q) ∧ m ≤ (bb+s+q)-(ba+s+p)) := by
  unfold interaction at h
  omega

end PairedTranslationChoice
