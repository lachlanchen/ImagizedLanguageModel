import Std

namespace SourceEditSupervision

/-- Loss for two exposures of the correct stopped answer on each pictured page. -/
def aligned (aa bb : Int) := 2 * (aa + bb)

/-- Loss for exposing both answers on each of the two pictured pages. -/
def independent (aa ab ba bb : Int) := aa + ab + ba + bb

/-- The entire difference is the sum of the two conditional preference margins.
The four entries can be fixed-point negative log probabilities. -/
theorem pairedRiskDifference (aa ab ba bb : Int) :
    independent aa ab ba bb - aligned aa bb = (ab - aa) + (ba - bb) := by
  unfold aligned independent
  omega

/-- With no access to the pictured edit, both training assignments have equal risk. -/
theorem blindEqual (a b : Int) : independent a b a b = aligned a b := by
  unfold aligned independent
  omega

/-- A binary squared-error version. Scores u and v predict the first assignment
on the first and second page, in the same integer units as the target scale s. -/
theorem brierGain (s u v : Int) :
    ((s-u)*(s-u) + u*u + (s-v)*(s-v) + v*v) -
      2*((s-u)*(s-u) + v*v) = 2*(s*(u-v)) := by
  simp only [Int.sub_mul, Int.mul_sub, Int.mul_comm u s, Int.mul_comm v s]
  omega

/-- An aggregate margin can improve even if the second page is still misread. -/
example : aligned 1 4 < independent 1 8 2 4 ∧ (2:Int) < 4 := by decide

#print axioms pairedRiskDifference
#print axioms blindEqual
#print axioms brierGain
end SourceEditSupervision
