import Std

namespace FactAnswerCapacity

/-- One row of a frozen writer's responses to the complete candidate list.
The selected index is characterized by membership; repeated spans are allowed. -/
structure Trial where
  offered : List Bool
  chosen : Bool
  valid : chosen ∈ offered

def capacity (t : Trial) : Bool := t.offered.contains true

theorem success_requires_capacity (t : Trial) (h : t.chosen = true) :
    capacity t = true := by
  have hm := t.valid
  rw [h] at hm
  simpa [capacity] using hm

def count (f : Trial → Bool) : List Trial → Nat
  | [] => 0
  | t :: ts => (if f t then 1 else 0) + count f ts

/-- Every error is either outside the support of the frozen system or lies in
its support but is missed by the selector. These two terms can be measured. -/
theorem exact_error_budget (ts : List Trial) :
    count Trial.chosen ts + count (fun t => capacity t && !t.chosen) ts +
      count (fun t => !capacity t) ts = ts.length := by
  induction ts with
  | nil => simp [count]
  | cons t ts ih =>
    have adequate := success_requires_capacity t
    cases hc : t.chosen <;> cases hs : capacity t <;>
      simp_all [count] <;> omega

end FactAnswerCapacity
