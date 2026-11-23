import Std

namespace VisualExpertLearning

/-- At each labelled page the predictors that emit the accepted physical answer
survive. On a mistake, a different predicted spelling had at least that many
votes. This argument allows any finite output alphabet. -/
inductive Trace : Nat → Nat → Nat → Prop
  | done (n : Nat) : Trace n n 0
  | hit {n kept last m : Nat} (subset : kept ≤ n) (rest : Trace kept last m) :
      Trace n last m
  | miss {n kept rival last m : Nat} (disjoint : kept + rival ≤ n)
      (plurality : kept ≤ rival) (rest : Trace kept last m) : Trace n last (m+1)

/-- On a miss, two disjoint candidate groups account for at least twice the
number of consistent survivors. Each further miss compounds the reduction. -/
theorem survivor_bound {n last m : Nat} (history : Trace n last m) :
    2^m * last ≤ n := by
  induction history with
  | done n => simp
  | hit subset _ ih => exact Nat.le_trans ih subset
  | @miss n kept rival last m disjoint plurality _ ih =>
    have half : 2 * kept ≤ n := by omega
    calc
      2^(m+1) * last = 2 * (2^m * last) := by
        rw [Nat.pow_succ]
        ac_rfl
      _ ≤ 2 * kept := Nat.mul_le_mul_left 2 ih
      _ ≤ n := half

theorem realizable_mistakes {n last m : Nat} (history : Trace n last m)
    (witness : 1 ≤ last) : 2^m ≤ n := by
  have h := Nat.mul_le_mul_left (2^m) witness
  simpa using Nat.le_trans h (survivor_bound history)

#print axioms realizable_mistakes
end VisualExpertLearning
