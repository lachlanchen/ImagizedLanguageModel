import Std

namespace OrderedSourcePlan

inductive Action (G K : Type) where
  | draw : G → Action G K
  | copy : K → Action G K

inductive Run {G K : Type} : List K → List (Action G K) → Prop where
  | stop : Run [] []
  | draw {ks acts} (g : G) : Run ks acts → Run ks (.draw g :: acts)
  | copy {ks acts} (k : K) : Run ks acts → Run (k :: ks) (.copy k :: acts)

def pointers {G K : Type} : List (Action G K) → List K
  | [] => []
  | .draw _ :: rest => pointers rest
  | .copy k :: rest => k :: pointers rest

/-- A completed accepting path realizes the whole ordered predicted field list.
There are no extra source pointers. Arbitrarily many ordinary glyphs may occur
between pointers and after the last one. Conversely the constraint excludes
no sentence whose pointer sequence is exactly the plan. -/
theorem accepts_iff {G K : Type} (ks : List K) (acts : List (Action G K)) :
    Run ks acts ↔ pointers acts = ks := by
  constructor
  · intro h
    induction h with
    | stop => rfl
    | draw _ _ ih => exact ih
    | copy k _ ih => exact congrArg (List.cons k) ih
  · intro h
    subst ks
    induction acts with
    | nil => exact Run.stop
    | cons a rest ih =>
      cases a with
      | draw g => exact Run.draw g ih
      | copy k => exact Run.copy k ih

/-- Enforcing a wrong inferred plan cannot repair it. This gives the
stage-specific ground-truth upper bound for literal pointer coverage. -/
theorem wrong_plan {G K : Type} {wanted chosen : List K} {acts : List (Action G K)}
    (wrong : chosen ≠ wanted) (done : Run chosen acts) : pointers acts ≠ wanted := by
  rw [(accepts_iff chosen acts).mp done]
  exact wrong

#print axioms accepts_iff
#print axioms wrong_plan
end OrderedSourcePlan
