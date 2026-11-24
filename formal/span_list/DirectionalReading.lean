import Std

namespace DirectionalReading

/-- A classifier using any exchange-invariant representation has one output
on both orders. If the correct roles differ, at most one can be recovered. -/
theorem exchange_error {A F Y : Type} (view : A → A → F) (predict : F → Y)
    (l r : A) (one two : Y) (same : view l r = view r l) (diff : one ≠ two) :
    ¬ (predict (view l r) = one ∧ predict (view r l) = two) := by
  rintro ⟨h₁,h₂⟩
  rw [same] at h₁
  exact diff (h₁.symm.trans h₂)

/-- An antisymmetric scalar feature can furnish both required margins
when the right context is separated from the left by that margin. -/
theorem signed_margin (l r margin : Int) (gap : l + margin ≤ r) :
    margin ≤ r - l ∧ l - r ≤ -margin := by omega

/-- A symmetric correction cancels from the difference between exchanged
contexts, even when the inherited optical reader already distinguishes them. -/
theorem inherited_difference (a b common : Int) :
    (a+common)-(b+common) = a-b := by omega

#print axioms exchange_error
#print axioms signed_margin
end DirectionalReading
