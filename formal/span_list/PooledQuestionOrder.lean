import Std

namespace PooledQuestionOrder
/-- Every local visual window makes an independent contribution. Reordering
    those windows leaves the sum and maximum of every output feature unchanged. -/
theorem pool_perm {α β : Type} {xs ys : List α} (p : xs.Perm ys) (f : α → β)
    (op : β → β → β) (z : β)
    (swap : ∀ a b c, op a (op b c) = op b (op a c)) :
    (xs.map f).foldr op z = (ys.map f).foldr op z := by
  induction p with
  | nil => rfl
  | cons x _ ih => simp [ih]
  | swap x y l => simp [swap (f y) (f x)]
  | trans _ _ ih jh => exact ih.trans jh

theorem same_query_cannot_switch {Page Q Answer : Type} (p : Page)
    (f : Page → Q → Answer) {x y : Q} (same : x = y) {a b : Answer}
    (different : a ≠ b) : ¬ (f p x = a ∧ f p y = b) := by
  subst y
  intro ⟨ha,hb⟩
  exact different (ha.symm.trans hb)
end PooledQuestionOrder
