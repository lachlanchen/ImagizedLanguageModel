import Std

namespace StyleOrder
variable {Ink State Reply : Type}

/-- A style summary built with commuting cell updates sees the same evidence
    after any permutation. The state can carry both a sum and its count. -/
theorem perm_invariant (step : Ink → State → State)
    (commute : ∀ a b s, step a (step b s) = step b (step a s))
    (initial : State) {a b : List Ink} (h : a.Perm b) :
    a.foldr step initial = b.foldr step initial := by
  induction h with
  | nil => rfl
  | cons x _ ih => simp only [List.foldr_cons]; rw [ih]
  | swap x y rest => simp only [List.foldr_cons]; apply commute
  | trans _ _ ih₁ ih₂ => exact ih₁.trans ih₂

/-- Hold page and source regions fixed inside the decoder. Two instructions
    that differ only in cell order cannot elicit two distinct required images. -/
theorem incompatible_replies (step : Ink → State → State)
    (commute : ∀ a b s, step a (step b s) = step b (step a s))
    (initial : State) (decode : State → Reply) {a b : List Ink}
    (p : a.Perm b) {u v : Reply} (distinct : u ≠ v) :
    ¬ (decode (a.foldr step initial) = u ∧ decode (b.foldr step initial) = v) := by
  intro ⟨hu, hv⟩
  exact distinct (hu.symm.trans ((congrArg decode (perm_invariant step commute initial p)).trans hv))

#print axioms incompatible_replies
end StyleOrder
