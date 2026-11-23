import Std

namespace PicturedQuestionSwitch

/-- A deterministic page-only writer cannot answer both questions on one page
when the two accepted sets of stopped image replies are disjoint. -/
theorem both_require_change {Reply : Type} (accept₁ accept₂ : Reply → Prop)
    (disjoint : ∀ r, accept₁ r → ¬ accept₂ r) {r₁ r₂ : Reply}
    (first : accept₁ r₁) (second : accept₂ r₂) : r₁ ≠ r₂ := by
  rintro rfl
  exact disjoint r₁ first second

/-- Each completely correct switch witnesses a change of reply. This holds
for any finite collection, including a sample selected without reading outputs. -/
theorem count_witnesses (xs : List (Bool × Bool))
    (valid : ∀ x ∈ xs, x.1 = true → x.2 = true) :
    (xs.filter (fun x => x.1)).length ≤ (xs.filter (fun x => x.2)).length := by
  induction xs with
  | nil => decide
  | cons x xs ih =>
    have tail := ih (fun y hy => valid y (List.mem_cons_of_mem x hy))
    have head := valid x (List.mem_cons_self x xs)
    obtain ⟨a,b⟩ := x
    cases a <;> cases b <;> simp_all [List.filter] <;> omega

#print axioms both_require_change
#print axioms count_witnesses
end PicturedQuestionSwitch
