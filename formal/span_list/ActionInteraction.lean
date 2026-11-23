import Std

namespace ActionInteraction

/-- The entries are signed log evidence for the first name over the second.
Rows are questions and columns are the original and exchanged actions. -/
def interaction (x00 x01 x10 x11 : Int) := x00 - x01 - x10 + x11

theorem cancel_bias (a b c d q₀ q₁ w₀ w₁ : Int) :
    interaction (a+q₀+w₀) (b+q₀+w₁) (c+q₁+w₀) (d+q₁+w₁)
    = interaction a b c d := by unfold interaction; omega

theorem four_margins (a b c d m : Int)
    (h0 : m ≤ a) (h1 : b ≤ -m) (h2 : c ≤ -m) (h3 : m ≤ d) :
    4*m ≤ interaction a b c d := by unfold interaction; omega

/-- One very confident logit can mask three errors. A positive interaction
alone cannot qualify a reader. -/
theorem positive_with_three_wrong :
    0 < interaction 20 1 1 (-1) ∧ (0:Int) < 1 ∧ (-1:Int) < 0 := by decide

#print axioms cancel_bias
#print axioms four_margins
end ActionInteraction
