import Std
namespace VisualActionDescription
/-- Two measured integer scores have a stable ordering when both a change
of wording and an approximation by an optical student are small enough. -/
theorem two_stage (a b p q x y e d : Int)
    (margin : b + 2*(e+d) < a)
    (ha : a-e ≤ p) (hb : q ≤ b+e) (hp : p-d ≤ x) (hq : y ≤ q+d) : y < x := by omega

/-- A question encoder collapsing descriptions that require opposite rows
cannot give both correct answers, even if the source encoder is exact. -/
theorem need_distinct {Q R : Type} (f : Q → R) (choose : R → Bool)
    (x y : Q) (same : f x = f y) : ¬(choose (f x) = true ∧ choose (f y) = false) := by
  rw [same]
  simp
/-- Once a line is fixed, a question-independent object pointer gives a single
position. Two distinct correct positions in that line cannot both be returned. -/
theorem single_object_limit {Line Position : Type} (pointer : Line → Position)
    (line : Line) {first second : Position} (different : first ≠ second) :
    ¬ (pointer line = first ∧ pointer line = second) := by
  intro h
  exact different (h.1.symm.trans h.2)
#print axioms single_object_limit
#print axioms two_stage
#print axioms need_distinct
end VisualActionDescription
