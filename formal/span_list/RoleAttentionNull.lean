import Std

namespace RoleAttentionNull

def total (n : Nat) (f : Nat → Nat) : Nat :=
  match n with
  | 0 => 0
  | m + 1 => total m f + f m

theorem total_add (n : Nat) (f g : Nat → Nat) :
    total n (fun j => f j + g j) = total n f + total n g := by
  induction n with
  | zero => rfl
  | succ n ih => simp only [total, ih]; omega

theorem total_scale (n c : Nat) (f : Nat → Nat) :
    total n (fun j => c * f j) = c * total n f := by
  induction n with
  | zero => simp [total]
  | succ n ih => simp [total, ih, Nat.mul_add]

theorem total_zero (n : Nat) : total n (fun _ => 0) = 0 := by
  induction n <;> simp_all [total]

/-- Fix the attention on tagged cells at any common rational resolution.
For every cell, a balanced set of assignments gives the focal role to that
cell on exactly k assignments. Summing the matching mass over the assignments
then gives k times the entire tagged mass. This allows nonuniform attention
and arbitrary dependence between the roles assigned to different cells.
Divide both sides by the number of assignments to obtain the null mean. -/
theorem balanced_assignment_mass (cells assignments k : Nat)
    (ink : Nat → Nat) (marked : Nat → Nat → Nat)
    (balanced : ∀ i, i < cells → total assignments (fun j => marked j i) = k) :
    total assignments (fun j => total cells (fun i => ink i * marked j i))
      = total cells ink * k := by
  induction cells with
  | zero => simp [total, total_zero]
  | succ cells ih =>
    have prior := ih (fun i hi => balanced i (Nat.lt_trans hi (Nat.lt_succ_self cells)))
    have last := balanced cells (Nat.lt_succ_self cells)
    simp only [total]
    rw [total_add, prior, total_scale, last, Nat.add_mul]

end RoleAttentionNull
