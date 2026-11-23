import Std

namespace IntervalTransfer

/-- A physical spelling can occur at several disjoint page intervals. -/
def mass (rows : List (Nat × Nat)) (answer : Nat) : Nat :=
  match rows with
  | [] => 0
  | (label,w) :: rest => (if label = answer then w else 0) + mass rest answer

def total : List (Nat × Nat) → Nat
  | [] => 0
  | (_,w) :: rest => w + total rest

theorem rival_le_complement (rows : List (Nat × Nat)) (gold rival : Nat)
    (distinct : gold ≠ rival) :
    mass rows rival + mass rows gold ≤ total rows := by
  induction rows with
  | nil => simp [mass,total]
  | cons row rest ih =>
    obtain ⟨label, w⟩ := row
    by_cases hg : label = gold <;> by_cases hr : label = rival <;>
      simp_all [mass,total] <;> omega

theorem majority_wins (rows : List (Nat × Nat)) (gold rival : Nat)
    (h : total rows < 2 * mass rows gold) (d : gold ≠ rival) :
    mass rows rival < mass rows gold := by
  have bound := rival_le_complement rows gold rival d
  omega

/-- The highest single location has label 1, while total mass favours label 0. -/
example : 3 < 5 ∧ mass [(0,3),(1,5),(0,3)] 1 < mass [(0,3),(1,5),(0,3)] 0 := by decide

/-- Exact end-to-end accuracy improves precisely when the writer repairs more
reader misses than it spoils exact selections, on the identical packet. -/
theorem writer_gain (kept lost repaired : Nat) :
    kept + lost < kept + repaired ↔ lost < repaired := by omega

#print axioms majority_wins
#print axioms writer_gain
end IntervalTransfer
