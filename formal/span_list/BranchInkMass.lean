import Std
namespace BranchInkMass

/-- Weighted, compatible continuations after an observed prefix. A completed
reply contributes its final stop as one ordinary choice here. Pad is excluded. -/
def weight {α : Type} : List (α × Nat) → Nat
  | [] => 0
  | (_,w) :: ps => w + weight ps

def mass {α : Type} [DecidableEq α] (g : α) : List (α × Nat) → Nat
  | [] => 0
  | (c,w) :: ps => (if c = g then w else 0) + mass g ps

theorem disjoint_mass {α : Type} [DecidableEq α] (a b : α) (h : a ≠ b)
    (paths : List (α × Nat)) : mass a paths + mass b paths ≤ weight paths := by
  induction paths with
  | nil => simp [mass, weight]
  | cons p ps ih =>
    obtain ⟨c,w⟩ := p
    by_cases ha : c = a <;> by_cases hb : c = b
    · exact False.elim (h (ha.symm.trans hb))
    all_goals simp [mass, weight, ha, hb, h, Ne.symm h] <;> omega

/-- If any other output ties or beats the requested next glyph, at most half
the compatible prefix mass goes to that glyph. When prefix mass is positive
this implies log(H/G) >= log 2 (infinite when G=0). The converse need not hold. -/
theorem mistake_half {α : Type} [DecidableEq α] (a b : α) (h : a ≠ b)
    (paths : List (α × Nat)) (beats : mass a paths ≤ mass b paths) :
    2 * mass a paths ≤ weight paths := by
  have hd := disjoint_mass a b h paths
  omega
end BranchInkMass
