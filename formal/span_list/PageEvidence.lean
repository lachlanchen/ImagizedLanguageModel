import Std
namespace PageEvidence

/-- A target on pages factors through a page observation exactly when it agrees
on each observable fibre. The section selects one page for each attained view. -/
theorem attainable_factorization {Page View Reply : Type}
    (observe : Page → View) (target : Page → Reply)
    (representative : {v : View // ∃ p, observe p = v} → Page)
    (section_ok : ∀ v, observe (representative v) = v.val) :
    (∃ write : {v : View // ∃ p, observe p = v} → Reply,
      ∀ p, write ⟨observe p, p, rfl⟩ = target p) ↔
    (∀ p q, observe p = observe q → target p = target q) := by
  constructor
  · rintro ⟨write, correct⟩ p q same
    rw [← correct p, ← correct q]
    exact congrArg write (Subtype.ext same)
  · intro consistent
    exact ⟨fun v => target (representative v), fun p =>
      consistent _ p (section_ok ⟨observe p, p, rfl⟩)⟩

/-- Giving each subject the same observation before and after the exchange
bounds exact accuracy on a board even when the two subjects are distinguishable. -/
theorem unchanged_year_bound {Image : Type} [DecidableEq Image]
    (a b a₁ a₂ b₁ b₂ : Image) (ha : a₁ ≠ a₂) (hb : b₁ ≠ b₂) :
    (if a = a₁ then 1 else 0) + (if a = a₂ then 1 else 0) +
    (if b = b₁ then 1 else 0) + (if b = b₂ then 1 else 0) ≤ (2 : Nat) := by
  by_cases a1 : a = a₁ <;> by_cases a2 : a = a₂ <;>
    by_cases b1 : b = b₁ <;> by_cases b2 : b = b₂ <;> simp_all

#print axioms attainable_factorization
#print axioms unchanged_year_bound
end PageEvidence
