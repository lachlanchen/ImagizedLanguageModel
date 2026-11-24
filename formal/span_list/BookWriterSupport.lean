import Std

namespace BookWriterSupport

/-- A route may emit a palette bitmap or a bitmap already in the visible history.
The history is stored newest first. This describes support, before selecting a mode. -/
def Reach (bank : G → Prop) (history : List G) : List G → Prop
  | [] => True
  | y :: ys => (bank y ∨ y ∈ history) ∧ Reach bank (y :: history) ys

/-- Self-feedback cannot introduce another shape. Conversely every finite target
over the initial union is reachable with positive-mass palette and copy routes
when the prompt remains visible. This is a support theorem, independent of fit. -/
theorem reachable_iff (bank : G → Prop) (h target : List G) :
    Reach bank h target ↔ ∀ g ∈ target, bank g ∨ g ∈ h := by
  induction target generalizing h with
  | nil => simp [Reach]
  | cons y ys ih =>
    rw [Reach, ih, List.forall_mem_cons]
    constructor
    · rintro ⟨first, rest⟩
      refine ⟨first, ?_⟩
      intro g mem
      rcases rest g mem with b | seen
      · exact Or.inl b
      · rcases List.mem_cons.mp seen with eq | orig
        · exact eq ▸ first
        · exact Or.inr orig
    · rintro ⟨first, rest⟩
      refine ⟨first, ?_⟩
      intro g mem
      rcases rest g mem with b | orig
      · exact Or.inl b
      · exact Or.inr (List.mem_cons_of_mem _ orig)

end BookWriterSupport
