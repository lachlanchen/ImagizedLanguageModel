import Std

namespace OpticalCandidateCeiling

/-- An evaluator accepts a physical strip if its pixels match any human reference.
The witness can choose an accepted candidate whenever one is available. -/
def witness {A : Type} (accept : A → Bool) : List A → Option A
  | [] => none
  | a :: tail => if accept a then some a else witness accept tail

theorem witness_exact {A : Type} (accept : A → Bool) (xs : List A) :
    ((witness accept xs).map accept).getD false = xs.any accept := by
  induction xs with
  | nil => rfl
  | cons a as ih =>
    cases e : accept a <;> simp [witness, e, ih]

/-- Sum the two sources of inevitable or removable error. The first argument
of each observation is physical availability, the second is emitted credit. -/
def totals (f : Bool × Bool → Nat) : List (Bool × Bool) → Nat
  | [] => 0
  | p :: tail => f p + totals f tail

theorem errors_partition (xs : List (Bool × Bool))
    (valid : ∀ p ∈ xs, p.2 = true → p.1 = true) :
    totals (fun p => if p.2 then 0 else 1) xs =
    totals (fun p => if p.1 then 0 else 1) xs +
    totals (fun p => if p.1 && !p.2 then 1 else 0) xs := by
  induction xs with
  | nil => rfl
  | cons p ps ih =>
    have h := ih (fun q hq => valid q (List.mem_cons_of_mem p hq))
    have hp := valid p (by simp)
    rcases p with ⟨a, b⟩
    cases a <;> cases b <;> simp_all [totals] <;> omega

end OpticalCandidateCeiling
