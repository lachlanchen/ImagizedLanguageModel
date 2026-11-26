import Std

namespace VisualActionBudget

/-- Cost in units of one score per legal action. K accounts for computation
shared by the alternatives at a decision; a stop is also a decision. -/
def cost (decisions shared choices : Nat) := decisions * (shared + choices)

/-- When strips save s decisions but add v legal choices, the exact break-even
condition is that removed work exceed additional softmax work. No assumption
on token independence, perplexity or the semantics of the predicted string. -/
theorem cheaper_iff (remaining saved shared atoms extra : Nat) :
    cost remaining shared (atoms + extra) < cost (remaining + saved) shared atoms
      ↔ remaining * extra < saved * (shared + atoms) := by
  unfold cost
  rw [← Nat.add_assoc, Nat.mul_add, Nat.add_mul]
  omega

/-- The same accounting can be added across directions with unequal visual
repertoires, retaining exactly the extra scores due to the wider action head. -/
theorem gap_identity (remaining saved shared atoms extra : Nat) :
    cost remaining shared (atoms + extra) + saved * (shared + atoms) =
    cost (remaining + saved) shared atoms + remaining * extra := by
  unfold cost
  rw [← Nat.add_assoc, Nat.mul_add, Nat.add_mul]
  omega

/-- A finite visual strip vocabulary with every singleton preserves the
original physical writer's entire sequence support, including white glyphs. -/
theorem singleton_fallback {α : Type} (glyphs : List α) (legal : List α → Prop)
    (atoms : ∀ a ∈ glyphs, legal [a]) :
    ∃ actions : List (List α), actions.join = glyphs ∧ ∀ s ∈ actions, legal s := by
  refine ⟨glyphs.map (fun a => [a]), ?_, ?_⟩
  · induction glyphs with
    | nil => rfl
    | cons a tail ih => simp_all
  · intro s hs
    obtain ⟨a, ha, rfl⟩ := List.mem_map.mp hs
    exact atoms a ha

end VisualActionBudget
