import Std

namespace VisualFieldOrder

def render {α : Type} (flip : Bool) (fields : α × α) : α × α :=
  if flip then (fields.2, fields.1) else fields

def available {α : Type} (x target : α × α) : Prop :=
  x = target ∨ (x.2, x.1) = target

theorem single_exact {α : Type} (x t : α × α) :
    (∃ f, render f x = t) ↔ available x t := by
  constructor
  · rintro ⟨f,h⟩
    cases f with
    | false => exact Or.inl h
    | true => exact Or.inr h
  · rintro (h|h)
    · exact ⟨false,h⟩
    · exact ⟨true,h⟩

/-- For two different instructions on one page, this ceiling is tight even
    when the original selector gives different values on the two requests. -/
theorem two_requests {α : Type} (x y : α × α) (a b : α) :
    (∃ f g, render f x = (a,b) ∧ render g y = (b,a)) ↔
      available x (a,b) ∧ available y (b,a) := by
  constructor
  · rintro ⟨f,g,h,k⟩
    exact ⟨(single_exact ..).mp ⟨f,h⟩,(single_exact ..).mp ⟨g,k⟩⟩
  · rintro ⟨h,k⟩
    obtain ⟨f,hf⟩ := (single_exact ..).mpr h
    obtain ⟨g,hg⟩ := (single_exact ..).mpr k
    exact ⟨f,g,hf,hg⟩

theorem order_blind {α : Type} (a b : α) (different : a ≠ b) (answer : α × α) :
    ¬ (answer = (a,b) ∧ answer = (b,a)) := by
  rintro ⟨h,k⟩
  exact different (congrArg Prod.fst (h.symm.trans k))

#print axioms two_requests
#print axioms order_blind
end VisualFieldOrder
