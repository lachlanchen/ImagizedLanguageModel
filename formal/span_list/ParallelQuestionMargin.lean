import Std

namespace ParallelQuestionMargin

/-- Logit gap for choosing the first translation over the second. A shared
passage-dependent offset cancels in every gap. -/
def gap (a b : Int) := a - b

theorem offset (a b k : Int) : gap (a+k) (b+k) = gap a b := by
  unfold gap
  omega

/-- Opposite questions cannot both meet a positive contrastive margin when
their pictured-query representations collapse to the same scores. -/
theorem translation_separation (a b c d m : Int) (hm : 0 < m)
    (first : m ≤ gap a b) (second : m ≤ gap d c) :
    gap a b ≠ gap c d := by
  unfold gap at *
  omega

/-- Successful translation matching alone imposes no choice between answer
images: a constant answer decoder satisfies the same translation inequalities. -/
theorem matching_allows_constant {α : Type} (reply : α) :
    ∃ (a b c d : Int) (decode : Bool → α),
      0 < gap a b ∧ 0 < gap d c ∧ decode true = decode false := by
  exact ⟨1, 0, 0, 1, fun _ => reply, by decide, by decide, rfl⟩

#print axioms translation_separation
#print axioms matching_allows_constant
end ParallelQuestionMargin
