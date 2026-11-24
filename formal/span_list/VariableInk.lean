import Std

namespace VariableInk

inductive Act (G R : Type) where
  | draw : G → Act G R
  | copy : R → Act G R

def emit {G R : Type} (source : R → List G) : List (Act G R) → List G
  | [] => []
  | .draw g :: tail => g :: emit source tail
  | .copy r :: tail => source r ++ emit source tail

/-- Replacing every physical mark in the source and every literal by the
same map commutes with rendering for arbitrary alphabets and source spans. -/
theorem transport {G H R : Type} (f : G → H) (source : R → List G)
    (plan : List (Act G R)) :
    (emit source plan).map f =
      emit (fun r => (source r).map f)
        (plan.map (fun a => match a with
          | .draw g => Act.draw (f g)
          | .copy r => Act.copy r)) := by
  induction plan with
  | nil => rfl
  | cons action tail ih =>
    cases action <;> simp [emit, List.map_append, ih]

/-- A new or unencoded glyph can only come from a literal drawing or a
copied input span. The statement also covers repeated and empty spans. -/
theorem support {G R : Type} (g : G) (src : R → List G) (p : List (Act G R)) :
    g ∈ emit src p ↔ Act.draw g ∈ p ∨ ∃ r, Act.copy r ∈ p ∧ g ∈ src r := by
  induction p with
  | nil => simp [emit]
  | cons a t ih =>
    cases a with
    | draw h => simp [emit, ih, or_assoc]
    | copy r =>
      have separate : (Act.draw g : Act G R) ≠ Act.copy r := by intro h; cases h
      simp only [emit, List.mem_append, ih, List.mem_cons, separate, false_or, Act.copy.injEq]
      constructor
      · rintro (hr | hl | ⟨s, hs, he⟩)
        · exact Or.inr ⟨r, Or.inl rfl, hr⟩
        · exact Or.inl hl
        · exact Or.inr ⟨s, Or.inr hs, he⟩
      · rintro (hl | ⟨s, (rfl | hs), he⟩)
        · exact Or.inr (Or.inl hl)
        · exact Or.inl he
        · exact Or.inr (Or.inr ⟨s, hs, he⟩)

/-- If a representation merges two queries on the same pictured card, any
deterministic reader of that representation misses at least one of their
different target fields.  The premise can be tested with equal-multiset queries. -/
theorem distinctFieldNeedsDistinctView {Q Z K : Type} (observe : Q → Z)
    (readField : Z → K) (a b : Q) (left right : K)
    (same : observe a = observe b) (different : left ≠ right) :
    ¬ (readField (observe a) = left ∧ readField (observe b) = right) := by
  rintro ⟨ha, hb⟩
  apply different
  exact ha.symm.trans ((congrArg readField same).trans hb)

end VariableInk
