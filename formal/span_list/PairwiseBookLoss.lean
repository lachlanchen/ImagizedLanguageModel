import Std

namespace PairwiseBookLoss

/-- Nonnegative integer masses in a common unit; stop is included in both.
The candidate's conditional share among these two replies is p/(p+q).
A reversed or tied strict preference has at least one half rival share. -/
theorem reversed_iff (p q : Nat) : p ≤ q ↔ p + q ≤ 2*q := by omega

/-- At a fixed denominator this counts how many comparisons could rank
the wrong reply at least as high as the accepted one. -/
def errors (scale : Nat) : List Nat → Nat
  | [] => 0
  | q :: xs => if scale ≤ 2*q then errors scale xs + 1 else errors scale xs

def mass : List Nat → Nat
  | [] => 0
  | x :: xs => x + mass xs

theorem empirical (scale : Nat) (rivals : List Nat) :
    errors scale rivals * scale ≤ 2 * mass rivals := by
  induction rivals with
  | nil => simp [errors, mass]
  | cons a xs ih =>
    by_cases h : scale ≤ 2*a
    · simp only [errors, mass, if_pos h, Nat.succ_mul, Nat.mul_add]
      omega
    · simp only [errors, mass, if_neg h, Nat.mul_add]
      omega

/-- Winning a fixed comparison can coexist with an incorrect free choice. -/
example : (3:Nat) > 2 ∧ (4:Nat) > 3 := by decide

#print axioms reversed_iff
#print axioms empirical
end PairwiseBookLoss
