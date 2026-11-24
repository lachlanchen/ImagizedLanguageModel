import Std

namespace OpticalEdgeMass

/-- A path visits a given position-length edge at most once because glyph
positions strictly advance. The second component holds the product of all
other factors, scaled to a common denominator. -/
def paths (weight : Nat) : List (Bool × Nat) → Nat
  | [] => 0
  | (used, v) :: rest => (if used then weight * v else v) + paths weight rest

def through : List (Bool × Nat) → Nat
  | [] => 0
  | (used,v) :: rest => (if used then v else 0) + through rest

theorem perturb (xs : List (Bool × Nat)) (w h : Nat) :
    paths (w+h) xs = paths w xs + h * through xs := by
  induction xs with
  | nil => simp [paths, through]
  | cons x xs ih =>
    rcases x with ⟨used,v⟩
    cases used <;> simp [paths, through, Nat.add_mul, Nat.mul_add, ih] <;> omega

/-- Positive competing path mass attenuates the exact relative gain from
increasing this edge compared with increasing every accepted path. -/
theorem fractional_gain (xs : List (Bool × Nat)) (w h : Nat)
    (other : w * through xs < paths w xs) (step : 0 < h) :
    w * (h * through xs) < h * paths w xs := by
  rw [Nat.mul_left_comm]
  exact Nat.mul_lt_mul_of_pos_left other step

end OpticalEdgeMass
