import Std

namespace RequestedPosition

/-- A two-mention question supplies referents in physical reading order.
True requests the first referent. A context classifier consumes a view
in which both referent spans have been replaced by the same ink. -/
def select {K : Type} (first second : K) (position : Bool) : K :=
  if position then first else second

/-- For each construction, once the normalized images and anchor positions
coincide, success on BOTH counterfactuals is exactly the success of ONE
positional decision. This explains why many label swaps in one frame
cannot be counted as independent observations of syntax generalization. -/
theorem paired_iff_position {K V : Type} (a b : K) (different : a ≠ b)
    (v swapped : V) (aligned : swapped = v) (score : V → Bool) (gold : Bool) :
    (select a b (score v) = select a b gold ∧
     select b a (score swapped) = select b a gold) ↔ score v = gold := by
  subst swapped
  cases h : score v <;> cases gold <;> simp [select, h, different, Ne.symm different]

/-- Exact transport of a positional decision also holds for unlabelled marks
and a many-to-one change of alphabet. Correct retrieval of the span and
generation of the rest of the sentence are separate hypotheses. -/
theorem mapped_selection {K L : Type} (f : K → L) (a b : K) (p : Bool) :
    f (select a b p) = select (f a) (f b) p := by
  cases p <;> rfl

#print axioms paired_iff_position
#print axioms mapped_selection

end RequestedPosition
