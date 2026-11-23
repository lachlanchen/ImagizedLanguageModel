import Std

/-! Ordered comparisons for a scorer selecting an ink span. Integer scores represent
quantized endpoint sums. Floating-point optimization is evaluated separately. -/
namespace ParticipantMargin

def mistake (target rival : Int) : Int := if rival < target then 0 else 1
def hinge (target rival : Int) : Int := max 0 (1 + rival - target)

theorem pair_bound (target rival : Int) : mistake target rival ≤ hinge target rival := by
  unfold mistake hinge
  split <;> omega

def total (loss : Int → Int → Int) : List (Int × Int) → Int
  | [] => 0
  | (t,r) :: xs => loss t r + total loss xs

theorem corpus_bound (scores : List (Int × Int)) :
    total mistake scores ≤ total hinge scores := by
  induction scores with
  | nil => simp [total]
  | cons x xs ih =>
    obtain ⟨t,r⟩ := x
    exact Int.add_le_add (pair_bound t r) ih

/-- Pair supervision alone does not rule out a third, unlabelled winner. -/
theorem third_span : hinge 2 0 = 0 ∧ mistake 2 3 = 1 := by decide

/-- A joint start/end decoder may choose any maximum. Strict separation from
every other admissible span ensures that even tie-breaking cannot change the answer. -/
theorem recover {Span : Type} (valid : Span → Prop) (score : Span → Int)
    (truth chosen : Span) (ht : valid truth) (hc : valid chosen)
    (top : ∀ x, valid x → score x ≤ score chosen)
    (separate : ∀ x, valid x → x ≠ truth → score x < score truth) :
    chosen = truth := by
  apply Classical.byContradiction
  intro error
  have h := separate chosen hc error
  have k := top truth ht
  omega

/-- When a picture is edited, score differences, not absolute scores or physical
indices, must be preserved. A glyph insertion can move every later endpoint. -/
theorem edited_recovery {A B : Type} (valid : A → Prop) (edited : B → Prop)
    (score : A → Int) (after : B → Int) (move : A → B) (truth : A) (chosen : B)
    (ht : valid truth) (hc : edited chosen) (maps : ∀ x, valid x → edited (move x))
    (covers : ∀ y, edited y → ∃ x, valid x ∧ move x = y)
    (gap : ∀ x, valid x → after (move x) - after (move truth) = score x - score truth)
    (strict : ∀ x, valid x → x ≠ truth → score x < score truth)
    (best : ∀ y, edited y → after y ≤ after chosen) : chosen = move truth := by
  apply recover edited after (move truth) chosen (maps truth ht) hc best
  intro y hy neq
  obtain ⟨x,hx,rfl⟩ := covers y hy
  have diff : x ≠ truth := fun e => neq (congrArg move e)
  have h := strict x hx diff
  have k := gap x hx
  omega

#print axioms pair_bound
#print axioms corpus_bound
#print axioms third_span
#print axioms recover
#print axioms edited_recovery
end ParticipantMargin
