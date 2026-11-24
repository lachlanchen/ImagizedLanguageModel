import Std

namespace GreedyInkPrefix

/-- A deterministic optical decoder, unrolled for a fixed number of cells.
The history is newest first. A target can include its final physical stop. -/
def emit {α : Type} (choose : List α → α) (history : List α) : Nat → List α
  | 0 => []
  | n + 1 => let c := choose history; c :: emit choose (c :: history) n

def follows {α : Type} (choose : List α → α) (history : List α) : List α → Prop
  | [] => True
  | c :: tail => choose history = c ∧ follows choose (c :: history) tail

/-- Each conditional decision on an accepted target prefix must win against the
entire output space. Checking the real greedy result and checking those decisions
are equivalent up to the target's length, including its terminal decision. -/
theorem full_prefix {α : Type} (choose : List α → α) (history target : List α) :
    emit choose history target.length = target ↔ follows choose history target := by
  induction target generalizing history with
  | nil => simp [emit, follows]
  | cons c cs ih =>
    simp only [List.length_cons, emit, List.cons.injEq, follows]
    constructor
    · rintro ⟨h, rest⟩
      rw [h] at rest
      exact ⟨h, (ih _).mp rest⟩
    · rintro ⟨h, rest⟩
      exact ⟨h, h ▸ (ih _).mpr rest⟩

/-- For any finite-precision scoring implementation whose chosen cell attains
the global maximum, a positive margin against every distinct cell is sufficient.
The prefix equivalence above also handles ties through the actual tie rule. -/
theorem strict_winner {α : Type} (score : α → Int) (picked target : α)
    (maximal : ∀ a, score a ≤ score picked)
    (strict : ∀ a, a ≠ target → score a < score target) : picked = target := by
  apply Classical.byContradiction
  intro h
  have hm := maximal target
  have hs := strict picked h
  omega

end GreedyInkPrefix
