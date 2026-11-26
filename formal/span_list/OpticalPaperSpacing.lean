import Std

namespace OpticalPaperSpacing

/-- None denotes a completely blank, unmarked cell. Marked cells, including
marked whitespace, are retained as observations of type α. -/
def compact (previousBlank : Bool) : List (Option α) → List (Option α)
  | [] => []
  | none :: tail =>
    if previousBlank then compact true tail else none :: compact true tail
  | some glyph :: tail => some glyph :: compact false tail

/-- Inserting another blank into an existing run cannot affect any following
decision of a deterministic reader operating on the compact stream. -/
theorem duplicate_paper (p : Bool) (left right : List (Option α)) :
    compact p (left ++ none :: none :: right) =
    compact p (left ++ none :: right) := by
  induction left generalizing p with
  | nil => cases p <;> simp [compact]
  | cons x xs ih =>
    cases x <;> cases p <;> simp [compact, ih]

theorem decision_invariance (read : List (Option α) → β)
    (left right : List (Option α)) :
    read (compact false (left ++ none :: none :: right)) =
    read (compact false (left ++ none :: right)) :=
  congrArg read (duplicate_paper false left right)

end OpticalPaperSpacing
