import Std

namespace PhaseGlyphReadback

/-- An observation permits exact recovery of the address on the entire
specified support exactly when every observation fibre has one address.
The observation may be ink alone or (phase, ink). No encoding of the ink,
nor any probability distribution over the support, is assumed. -/
theorem exact_iff_constant {I O E : Type} [Inhabited E]
    (observe : I → O) (address : I → E) :
    (∃ read : O → E, ∀ i, read (observe i) = address i) ↔
    (∀ i j, observe i = observe j → address i = address j) := by
  letI (p : Prop) : Decidable p := Classical.propDecidable p
  constructor
  · rintro ⟨read, h⟩ i j eq
    rw [← h i, eq, h j]
  · intro single
    let read := fun o => if h : ∃ i, observe i = o then address (Classical.choose h) else default
    refine ⟨read, ?_⟩
    intro i
    have existsI : ∃ j, observe j = observe i := ⟨i, rfl⟩
    simp only [read, dif_pos existsI]
    exact single _ _ (Classical.choose_spec existsI)

theorem same_ink_conflict {I P G E : Type} [Inhabited E]
    (phase : I → P) (ink : I → G) (address : I → E)
    (unique_at_phase : ∀ i j, (phase i, ink i) = (phase j, ink j) → address i = address j)
    (i j : I) (same : ink i = ink j) (conflict : address i ≠ address j) :
    (∃ read : P × G → E, ∀ k, read (phase k, ink k) = address k) ∧
    ¬ (∃ read : G → E, ∀ k, read (ink k) = address k) := by
  constructor
  · exact (exact_iff_constant (fun k => (phase k, ink k)) address).2 unique_at_phase
  · intro h
    exact conflict ((exact_iff_constant ink address).1 h i j same)

end PhaseGlyphReadback
