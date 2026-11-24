import Std

namespace FactRankMass

def mass : List Nat → Nat
  | [] => 0
  | x :: xs => x + mass xs

/-- Nonnegative weights can be exact integer counts or quantized exponentials.
A list can contain several matching positions for the same visible answer. -/
theorem mass_le_count_times_max (xs : List Nat) (w : Nat)
    (h : ∀ x ∈ xs, x ≤ w) : mass xs ≤ xs.length * w := by
  induction xs with
  | nil => simp [mass]
  | cons x xs ih =>
    have hx := h x (by simp)
    have ht := ih (fun y hy => h y (by simp [hy]))
    simp only [mass, List.length_cons, Nat.succ_mul]
    omega

/-- If a wrong argmax is available, each matching position has at most its
weight, and competing mass includes its weight. The contrapositive certifies
a correct winner using only total accepted mass, rival mass and multiplicity. -/
theorem wrong_winner_bound (accepted : List Nat) (rival totalRival : Nat)
    (hm : ∀ x ∈ accepted, x ≤ rival) (ht : rival ≤ totalRival) :
    mass accepted ≤ accepted.length * totalRival := by
  exact Nat.le_trans (mass_le_count_times_max accepted rival hm)
    (Nat.mul_le_mul_left _ ht)

end FactRankMass
