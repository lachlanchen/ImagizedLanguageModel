import Std
namespace WriterContext

/-- A writer that receives only the same factual fields cannot obey conflicting obligations. -/
theorem incompatible_details {Page Fields Image : Type} (encode : Page → Fields)
    (write : Fields → Image) (a b : Page) (first second : Image → Prop)
    (same : encode a = encode b) (incompatible : ∀ y, first y → ¬second y) :
    ¬(first (write (encode a)) ∧ second (write (encode b))) := by
  rintro ⟨ha,hb⟩
  rw [← same] at hb
  exact incompatible _ ha hb

#print axioms incompatible_details
end WriterContext

