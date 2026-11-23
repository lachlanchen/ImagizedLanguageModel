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
    · rw [errors, if_pos h, Nat.add_mul, Nat.one_mul, mass, Nat.mul_add]
      omega
    · rw [errors, if_neg h, mass, Nat.mul_add]
      omega

/-- Winning a fixed comparison can coexist with an incorrect free choice. -/
example : (3:Nat) > 2 ∧ (4:Nat) > 3 := by decide

/-- Finite search certificate: a bound on the total unexamined mass can
certify a complete-answer winner even if only a few rivals were scored. -/
theorem member_le_mass {q : Nat} {xs : List Nat} (hq : q ∈ xs) : q ≤ mass xs := by
  induction xs with
  | nil => simp at hq
  | cons a xs ih =>
    rcases List.mem_cons.mp hq with h | h
    · subst a
      simp [mass]
    · have bound := ih h
      simp only [mass]
      omega

theorem dominates_all (gold : Nat) (seen unseen : List Nat)
    (wins : ∀ q ∈ seen, q < gold) (tail : mass unseen < gold) :
    ∀ q ∈ seen ++ unseen, q < gold := by
  intro q hq
  rcases List.mem_append.mp hq with h | h
  · exact wins q h
  · exact Nat.lt_of_le_of_lt (member_le_mass h) tail

#print axioms dominates_all
#print axioms reversed_iff
#print axioms empirical
end PairwiseBookLoss
