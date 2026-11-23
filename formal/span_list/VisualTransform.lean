import Std

namespace VisualTransform

/-- A reader that only observes augmented ink can recover every supplied
label exactly iff no observable image has two different training labels. -/
theorem exact_reader_iff {Example Image Label : Type} [Inhabited Label]
    (paint : Example → Image) (target : Example → Label) :
    (∃ read : Image → Label, ∀ x, read (paint x) = target x) ↔
      ∀ x y, paint x = paint y → target x = target y := by
  constructor
  · rintro ⟨read, correct⟩ x y same
    rw [← correct x, ← correct y, same]
  · intro consistent
    classical
    let read := fun image =>
      if h : ∃ x, paint x = image then target (Classical.choose h) else default
    refine ⟨read, ?_⟩
    intro x
    have h : ∃ y, paint y = paint x := ⟨x, rfl⟩
    simpa [read, dif_pos h] using consistent (Classical.choose h) x (Classical.choose_spec h)

#print axioms exact_reader_iff
end VisualTransform
