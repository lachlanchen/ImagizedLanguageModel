import Std

namespace OrderedGlyphPair

/-- This branch-and-bound certificate concerns the model's own ordered joint
distribution. Work in any common fixed-point log-score scale, and take the
cutoff from the largest omitted first-cell score. -/
theorem certified (first : A → Int) (second : A → B → Int)
    (retained : A → Prop) (chosen : A) (ending : B) (cutoff : Int)
    (normalized : ∀ a b, second a b ≤ 0)
    (bestRetained : ∀ a b, retained a →
      first a + second a b ≤ first chosen + second chosen ending)
    (omittedBound : ∀ a, ¬retained a → first a ≤ cutoff)
    (certificate : cutoff ≤ first chosen + second chosen ending)
    (a : A) (b : B) :
    first a + second a b ≤ first chosen + second chosen ending := by
  by_cases h : retained a
  · exact bestRetained a b h
  · have h₁ := omittedBound a h
    have h₂ := normalized a b
    omega

end OrderedGlyphPair
