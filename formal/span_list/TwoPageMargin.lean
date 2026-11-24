import Std

namespace TwoPageMargin

/-- Log odds use a common physical answer ordering. Page one supports its
first answer and page two supports its second. Scores may be in any fixed
integer units, including rounded log likelihoods. -/
theorem both_iff_difference_beats_bias (left right : Int) :
    (0 < left ∧ right < 0) ↔
    max (left + right) (-(left + right)) < left - right := by
  rw [Int.max_def]
  split <;> omega

/-- The smaller signed margin is the exact bottleneck for both decisions. -/
theorem weaker_positive (left right : Int) :
    0 < min left (-right) ↔ 0 < left ∧ right < 0 := by
  rw [Int.min_def]
  split <;> omega

end TwoPageMargin
