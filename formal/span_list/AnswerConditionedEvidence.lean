import Std

namespace AnswerConditionedEvidence

/-- With finite nonnegative weights w, total prior W and total joint mass Z,
posterior > prior for a supported span iff its response likelihood exceeds
the prior average. Compare the two ratios by cross multiplication. -/
theorem posterior_gain (w likelihood W Z : Nat) (supported : 0 < w) :
    (w * likelihood) * W > w * Z ↔ likelihood * W > Z := by
  rw [Nat.mul_assoc]
  exact Nat.mul_lt_mul_left supported

/-- Moving positive prior mass between two candidates strictly improves total
joint answer mass iff the destination has higher complete-answer likelihood. -/
theorem mass_transfer (rest units low high : Nat) (positive : 0 < units) :
    rest + units * high > rest + units * low ↔ high > low := by
  have h := @Nat.mul_lt_mul_left units low high positive
  omega

end AnswerConditionedEvidence
