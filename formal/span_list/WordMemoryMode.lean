import Std

namespace WordMemoryMode

/-- On a common denominator, exact mass is the resolved copy/memory/greedy
mass plus the unresolved spellings which draw this particular image. -/
def total (resolved omitted : Image → Nat) (x : Image) :=
  resolved x + omitted x

/-- A margin larger than the remaining spelling budget suffices even if all
omitted probability concentrates on one competing image outside the shortlist. -/
theorem uniqueMode (resolved omitted : Image → Nat) (winner : Image)
    (second budget : Nat)
    (runnerUp : ∀ x, x ≠ winner → resolved x ≤ second)
    (remaining : ∀ x, omitted x ≤ budget)
    (margin : second + budget < resolved winner) :
    ∀ x, x ≠ winner → total resolved omitted x < total resolved omitted winner := by
  intro x hx
  have bound := Nat.add_le_add (runnerUp x hx) (remaining x)
  unfold total
  omega

/-- Conversely, no tighter uniform guarantee can be deduced from these
bounds alone. If the margin test fails, all unseen mass can go to the rival. -/
theorem tight (first second budget : Nat) (insufficient : first ≤ second + budget) :
    ∃ resolved omitted : Bool → Nat,
      resolved true = first ∧ resolved false = second ∧
      omitted true + omitted false = budget ∧
      total resolved omitted true ≤ total resolved omitted false := by
  refine ⟨fun b => if b then first else second, fun b => if b then 0 else budget, ?_⟩
  simpa [total] using insufficient

end WordMemoryMode
