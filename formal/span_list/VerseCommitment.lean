import Std

namespace VerseCommitment

/-- Scores and completion masses use a common scale. The available action is a
pair of a hidden position and a proposed image. A winning score gives a sharp
two-error-unit guarantee for the first commitment under uniform calibration. -/
theorem firstMass (mass score : A → Int) (error : Int) (chosen : A)
    (upper : ∀ a, mass a ≤ score a + error)
    (lower : ∀ a, score a ≤ mass a + error)
    (best : ∀ a, score a ≤ score chosen) (other : A) :
    mass other ≤ mass chosen + 2 * error := by
  have h₁ := upper other
  have h₂ := lower chosen
  have h₃ := best other
  omega

/-- Counts of fully specified two-cell lines consistent with supplied ink.
Rows are the second position, columns the first. -/
def joint : Fin 2 → Fin 3 → Nat
  | 0, 0 => 21
  | 0, 1 => 19
  | 0, 2 => 16
  | 1, 0 => 29
  | 1, 1 => 7
  | 1, 2 => 8

def row (i : Fin 2) := joint i 0 + joint i 1 + joint i 2
def col (j : Fin 3) := joint 0 j + joint 1 j

/-- The most confident first commitment is the second cell's row zero. Its
best completion has mass 21. Starting at the left cell leads instead to mass
29. Perfectly estimated local confidences do not imply optimal line accuracy. -/
theorem twoCellCounterexample :
    (∀ j, col j < row 0) ∧ row 1 < row 0 ∧
    (∀ j, joint 0 j ≤ joint 0 0) ∧ (∀ j, col j ≤ col 0) ∧
    joint 0 0 < joint 1 0 ∧ joint 0 0 = 21 ∧ joint 1 0 = 29 := by decide

/-- In reciprocal passage scoring, any source-only or ending-only additive
preference cancels exactly. Scores can be scaled log likelihoods. -/
theorem cancelPreferences (aa ab ba bb x y u v : Int) :
    ((aa+x+u)-(ab+x+v)) + ((bb+y+v)-(ba+y+u)) =
      (aa-ab)+(bb-ba) := by omega

end VerseCommitment
