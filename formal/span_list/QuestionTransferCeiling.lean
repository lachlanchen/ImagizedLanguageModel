import Std

namespace QuestionTransferCeiling

/-- A grounded reply requires an acceptable image at the intended cell
and the pictured question selecting that cell. Both roles must succeed. -/
structure Trial where
  writable : Bool
  oldFocus : Bool
  newFocus : Bool

def complete (w f : Bool) : Nat := if w && f then 1 else 0

def counts (xs : List Trial) (f : Trial → Nat) : Nat := (xs.map f).sum

/-- With the writer fixed, complete gains come precisely from focus repairs
on writable pages, less new focus losses on that same subset. -/
theorem paired_balance (xs : List Trial) :
    counts xs (fun x => complete x.writable x.newFocus) +
      counts xs (fun x => complete x.writable (x.oldFocus && !x.newFocus)) =
    counts xs (fun x => complete x.writable x.oldFocus) +
      counts xs (fun x => complete x.writable (x.newFocus && !x.oldFocus)) := by
  induction xs with
  | nil => rfl
  | cons r rs ih =>
    rcases r with ⟨w, a, b⟩
    cases w <;> cases a <;> cases b <;>
      simp [counts, complete, List.sum_cons] at * <;> omega

theorem writer_ceiling (xs : List Trial) :
    counts xs (fun x => complete x.writable x.newFocus) ≤
    counts xs (fun x => complete x.writable true) := by
  induction xs with
  | nil => simp [counts]
  | cons r rs ih =>
    rcases r with ⟨w, a, b⟩
    cases w <;> cases b <;> simp [counts, complete, List.sum_cons] at * <;> omega

#print axioms paired_balance
#print axioms writer_ceiling
end QuestionTransferCeiling
