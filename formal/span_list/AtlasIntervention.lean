import Std

namespace AtlasIntervention

/-- A source intervention gives one labelled record a mark. Its output indicates
the positions at which an ordered copying plan reads that particular record. -/
def marked [DecidableEq α] (key : α) (plan : List α) : List Bool :=
  plan.map (fun k => decide (k = key))

/-- Singleton marks identify the order and multiplicity of every copied source.
Distinct headwords with identical images need not be distinguishable on the
unmodified book. A nonempty key type covers the empty-answer case as well. -/
theorem identify [DecidableEq α] (witness : α) (a b : List α)
    (h : ∀ k, marked k a = marked k b) : a = b := by
  induction a generalizing b with
  | nil =>
    cases b with
    | nil => rfl
    | cons z zs => simpa [marked] using h witness
  | cons x xs ih =>
    cases b with
    | nil => simpa [marked] using h witness
    | cons y ys =>
      have first := congrArg List.head? (h x)
      have eq : y = x := by
        by_cases eq : y = x
        · exact eq
        · simp [marked, eq] at first
      subst y
      have tail : ∀ k, marked k xs = marked k ys := by
        intro k
        exact List.cons.inj (h k) |>.2
      rw [ih ys tail]

end AtlasIntervention
