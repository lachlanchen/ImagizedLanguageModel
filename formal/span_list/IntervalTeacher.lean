import Std
namespace IntervalTeacher

/-- An occurrence carries positive or zero integer mass. Repeated intervals can
represent quantized distributions. Prefix comparison uses only physical cells. -/
def mass {α : Type} (xs : List (List α × Nat)) (f : List α → Bool) : Nat :=
  match xs with
  | [] => 0
  | (s,w) :: rest => (if f s then w else 0) + mass rest f

theorem stop_or_continue {α : Type} (xs : List (List α × Nat))
    (compatible : List α → Bool) (k : Nat) :
    mass xs compatible =
      mass xs (fun s => compatible s && decide (s.length = k)) +
      mass xs (fun s => compatible s && !(decide (s.length = k))) := by
  induction xs with
  | nil => rfl
  | cons entry rest ih =>
    obtain ⟨s,w⟩ := entry
    cases hs : compatible s <;> by_cases h : s.length = k <;>
      simp [mass,hs,h] at * <;> omega

/-- When compatible streams contain the observed prefix, the complement of
stopping is exactly existence of a next image at offset k. -/
theorem next_exists_iff {α : Type} (s : List α) (k : Nat) (h : k ≤ s.length) :
    s.length ≠ k ↔ k < s.length := by omega

#print axioms stop_or_continue
#print axioms next_exists_iff
end IntervalTeacher
