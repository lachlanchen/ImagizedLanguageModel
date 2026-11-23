import Std

namespace ActionPageInteraction

/-- Sign is the log-score of name A minus that of name B. Cells correspond
to page version (row) and requested action (column). -/
def interaction (x y z w : Int) := x - y - z + w

theorem cancel_separate_bias (x y z w p q a b : Int) :
    interaction (x+p+a) (y+p+b) (z+q+a) (w+q+b) = interaction x y z w := by
  unfold interaction
  omega

/-- A page bias plus a question bias cannot correctly assign both people
on both page versions, regardless of the scale or sign of either bias. -/
theorem additive_cannot_bind (p q a b : Int) :
    ¬ (p+a > 0 ∧ p+b < 0 ∧ q+a < 0 ∧ q+b > 0) := by omega

theorem four_correct (x y z w : Int) (h : x>0 ∧ y<0 ∧ z<0 ∧ w>0) :
    interaction x y z w ≥ 4 := by
  unfold interaction
  omega

example : interaction 7 (-2) 1 3 > 0 ∧ ¬ ((1:Int)<0) := by decide
#print axioms additive_cannot_bind
#print axioms four_correct
end ActionPageInteraction
