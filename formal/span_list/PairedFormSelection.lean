import Std

namespace PairedFormSelection

def credit [DecidableEq α] (prediction target : α) : Nat :=
  if prediction = target then 1 else 0

/-- For two different requested specimens on one fixed page, an answer that
ignores the changed request can earn at most one credit. This still holds if
the common answer consists of several glyphs or is an unterminated prefix. -/
theorem shared_answer_bound [DecidableEq α] (output a b : α) (h : a ≠ b) :
    credit output a + credit output b ≤ 1 := by
  by_cases left : output = a
  · have right : output ≠ b := by
      intro eq
      exact h (left.symm.trans eq)
    simp [credit, left, right, h, Ne.symm h]
  · by_cases right : output = b <;> simp [credit, left, right, h, Ne.symm h]

/-- If K pairs have two credits, the other pairs can supply at most N-K
additional credits. Thus a score above N certifies at least score-N pairs
with two correct, request-sensitive outputs. -/
theorem minimum_double_hits (both one missed : Nat) :
    2 * both + one - (both + one + missed) ≤ both := by omega

end PairedFormSelection
