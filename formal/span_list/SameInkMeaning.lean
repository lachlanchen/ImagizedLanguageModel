import Std

namespace SameInkMeaning
/-- The very same pictured headword has opposite reference readings in two
    contexts. A predictor that sees only the head cannot answer both. -/
theorem context_required {Ink Answer : Type} (head : Ink) (f : Ink → Answer)
    {a b : Answer} (distinct : a ≠ b) : ¬ (f head = a ∧ f head = b) := by
  intro ⟨ha, hb⟩
  exact distinct (ha.symm.trans hb)

/-- With independent random guesses from the same context-blind distribution,
    the probability of getting both distinct readings is at most one quarter.
    Counts are over equally likely samples; leftover mass goes to other answers. -/
theorem independent_pair_bound (a b other : Nat) :
    4 * a * b ≤ (a + b + other) * (a + b + other) := by
  have difference : 4 * a * b ≤ (a + b) * (a + b) := by
    rcases Nat.le_total a b with h | h
    · obtain ⟨d, hd⟩ := Nat.exists_eq_add_of_le h
      rw [hd]
      simp only [Nat.mul_add, Nat.add_mul, Nat.mul_assoc, Nat.mul_comm d a]
      omega
    · obtain ⟨d, hd⟩ := Nat.exists_eq_add_of_le h
      rw [hd]
      simp only [Nat.mul_add, Nat.add_mul, Nat.mul_assoc, Nat.mul_comm d b]
      omega
  exact Nat.le_trans difference (Nat.mul_le_mul (Nat.le_add_right _ _) (Nat.le_add_right _ _))
end SameInkMeaning
