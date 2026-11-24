import Std

namespace VisibleWriterSupport
abbrev supported (spell view : List Nat) (x : Nat) : Prop := x ∈ spell ∨ x ∈ view

/-- With positive routing probabilities on the allowed spelling and copy
branches, this describes exactly which new cells become representable. -/
theorem new_cell (spell strip page : List Nat) (x : Nat) :
    (supported spell (strip ++ page) x ∧ ¬ supported spell strip x) ↔
    (x ∈ page ∧ x ∉ spell ∧ x ∉ strip) := by
  simp only [supported, List.mem_append]
  constructor
  · rintro ⟨h, hn⟩
    have a : x ∉ spell := fun e => hn (Or.inl e)
    have b : x ∉ strip := fun e => hn (Or.inr e)
    exact ⟨(h.resolve_left a).resolve_left b, a, b⟩
  · rintro ⟨p,a,b⟩
    exact ⟨Or.inr (Or.inr p), fun h => h.elim a b⟩

/-- Every teacher-forced prefix has positive target support if all its glyphs
are available under the fixed view. Extending the view preserves this test. -/
theorem sequence_monotone (spell strip page answer : List Nat)
    (h : ∀ x ∈ answer, supported spell strip x) :
    ∀ x ∈ answer, supported spell (strip ++ page) x := by
  intro x hx
  rcases h x hx with hs | hv
  · exact Or.inl hs
  · exact Or.inr (List.mem_append.mpr (Or.inl hv))
end VisibleWriterSupport
