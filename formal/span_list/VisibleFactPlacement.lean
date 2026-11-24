import Std
namespace VisibleFactPlacement

/-- Non-overlapping physical placements in a specified left-to-right order.
The start/end offsets are measured in source cells, with exclusive end. -/
inductive Fits {α : Type} (occurs : α → Nat → Nat → Prop) :
    Nat → List α → Prop where
  | nil (p) : Fits occurs p []
  | cons {p s e w ws} : p ≤ s → occurs w s e → s < e →
      Fits occurs e ws → Fits occurs p (w :: ws)

theorem earlier {α : Type} {occurs : α → Nat → Nat → Prop}
    {p q : Nat} {ws : List α} (h : p ≤ q)
    (f : Fits occurs q ws) : Fits occurs p ws := by
  cases f with
  | nil => exact .nil _
  | cons hs ho he tail => exact .cons (Nat.le_trans h hs) ho he tail

/-- Choosing the earliest eligible finish for the next member of this
ordering preserves exactly the feasible continuations. Searching alternative
orders is still necessary. An interrupted order search cannot certify absence. -/
theorem earliest_finish {α : Type} {occurs : α → Nat → Nat → Prop}
    {p s e : Nat} {w : α} {ws : List α} (start : p ≤ s)
    (hit : occurs w s e) (positive : s < e)
    (minimal : ∀ a b, p ≤ a → occurs w a b → a < b → e ≤ b) :
    Fits occurs p (w :: ws) ↔ Fits occurs e ws := by
  constructor
  · intro f
    cases f with
    | cons hs ho he tail => exact earlier (minimal _ _ hs ho he) tail
  · exact fun tail => .cons start hit positive tail

#print axioms earliest_finish
end VisibleFactPlacement
