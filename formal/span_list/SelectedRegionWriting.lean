import Std

namespace SelectedRegionWriting

/-- Compare the same deterministic writer on a selected and an annotated image.
Outcomes may score any complete, stopped answer against multiple references.
The orientation is oracle credit minus autonomous credit. -/
def delta (gold free : Bool) : Int :=
  (if gold then 1 else 0) - (if free then 1 else 0)

def rescue (gold free : Bool) : Int := if gold && !free then 1 else 0
def spoil (gold free : Bool) : Int := if free && !gold then 1 else 0

theorem per_page (gold free : Bool) :
    delta gold free = rescue gold free - spoil gold free := by
  cases gold <;> cases free <;> decide

/-- If the two image inputs coincide and the writer is deterministic, their grades
coincide. Only cases with a changed region can contribute to an intervention. -/
theorem changed_only {X : Type} (writeGrade : X → Bool) (chosen oracle : X)
    (same : chosen = oracle) : delta (writeGrade oracle) (writeGrade chosen) = 0 := by
  subst chosen
  cases writeGrade oracle <;> decide

def sumCases (f : Bool → Bool → Int) : List (Bool × Bool) → Int
  | [] => 0
  | (g, p) :: rest => f g p + sumCases f rest

theorem intervention (cases : List (Bool × Bool)) :
    sumCases delta cases = sumCases rescue cases - sumCases spoil cases := by
  induction cases with
  | nil => rfl
  | cons h t ih =>
    rcases h with ⟨g, p⟩
    simp only [sumCases, per_page, ih]
    omega

end SelectedRegionWriting
