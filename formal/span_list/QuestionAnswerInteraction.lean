import Std

namespace QuestionAnswerInteraction

/-- Log-scores of two whole stopped answers under each of two pictured questions.
Integers represent any common fixed-point unit. The first answer belongs to the
first question, the second to the second question. -/
def interaction (aa ab ba bb : Int) : Int := aa + bb - ab - ba

/-- Every additive page-specific preference for either answer cancels, as does
every question-dependent normalization shared by the two answers. -/
theorem cancel_bias (aa ab ba bb rowA rowB colA colB : Int) :
    interaction (aa + rowA + colA) (ab + rowA + colB)
      (ba + rowB + colA) (bb + rowB + colB) = interaction aa ab ba bb := by
  unfold interaction
  omega

/-- Selecting the appropriate answer on both requests implies a positive
interaction. The converse need not hold. -/
theorem both_implies_positive (aa ab ba bb : Int)
    (a : ab < aa) (b : ba < bb) : 0 < interaction aa ab ba bb := by
  simp only [interaction]
  omega

example : 0 < interaction 7 3 3 2 ∧ ¬ ((3:Int) < 2) := by decide

#print axioms cancel_bias
#print axioms both_implies_positive
end QuestionAnswerInteraction
