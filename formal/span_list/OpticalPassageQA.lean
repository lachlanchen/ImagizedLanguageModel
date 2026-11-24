import Std

namespace OpticalPassageQA

/-- Endpoints and within-word trims address exactly the same cells as the flat
    half-open span. Intervening whitespace is retained in the answer. -/
theorem emitted_width (left right dropLeft dropRight : Nat)
    (valid : left + dropLeft + dropRight ≤ right) :
    (right - dropRight) - (left + dropLeft) + dropLeft + dropRight = right - left := by
  omega

/-- With disjoint sets of admissible answers, paired success certifies
    dependence on the pictured question even when many paraphrases are credited. -/
theorem paired_dependence {Q A : Type} (emit : Q → A) (p q : Q)
    (first second : A → Prop) (unique : ∀ a, first a → second a → False)
    (hp : first (emit p)) (hq : second (emit q)) : emit p ≠ emit q := by
  intro h
  rw [← h] at hq
  exact unique _ hp hq

/-- On a fixed page a pure span emitter cannot output a reference outside the
    image of the slice operation, whatever is learned by the span selector. -/
theorem answer_reachability {Q Interval A : Type} (pick : Q → Interval)
    (slice : Interval → A) (q : Q) (target : A)
    (unreachable : ∀ i, slice i ≠ target) : slice (pick q) ≠ target :=
  unreachable (pick q)

#print axioms emitted_width
#print axioms paired_dependence
#print axioms answer_reachability
end OpticalPassageQA
