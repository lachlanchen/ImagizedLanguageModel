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

/-- Coefficients of the four sign patterns, in common scaled units. The
aligned interaction must overcome the worst cell's additive name bias. -/
theorem four_signs_iff (c p q i : Int) :
    (c+p+q+i > 0 ∧ c+p-q-i < 0 ∧ c-p+q-i < 0 ∧ c-p-q+i > 0) ↔
    (i > -(c+p+q) ∧ i > c+p-q ∧ i > c-p+q ∧ i > -c+p+q) := by
  constructor <;> intro h <;> omega

/-- Without division, all margins reconstruct exactly from the four measured
cells. Transposing either axis flips the sign of the interaction. -/
theorem first_cell_reconstruction (x y z w : Int) :
    (x+y+z+w)+(x+y-z-w)+(x-y+z-w)+interaction x y z w = 4*x := by
  unfold interaction
  omega

/-- Smallest common signed-score increase needed for all four decisions to
clear a requested positive integer margin. The upper bound includes zero
because an already qualified interaction needs no increase. -/
def deficit (x y z w margin : Int) :=
  max 0 (max (margin-x) (max (margin+y) (max (margin+z) (margin-w))))

theorem uniform_interaction_repair (x y z w margin delta : Int) :
    deficit x y z w margin ≤ delta ↔
    0 ≤ delta ∧ margin ≤ x+delta ∧ y-delta ≤ -margin ∧
      z-delta ≤ -margin ∧ margin ≤ w+delta := by
  simp only [deficit, Int.max_le]
  omega

#print axioms uniform_interaction_repair
#print axioms four_signs_iff
#print axioms first_cell_reconstruction
example : interaction 7 (-2) 1 3 > 0 ∧ ¬ ((1:Int)<0) := by decide
#print axioms additive_cannot_bind
#print axioms four_correct
end ActionPageInteraction
