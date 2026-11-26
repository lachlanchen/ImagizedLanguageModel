import Std

namespace VisualSourcePaths

/-- Interaction in fixed-point units of target log probability, for any two
original/alternative base images and any two original/alternative source fields. -/
def cycle {B F : Type} (s : B → F → Int) (b a : B) (f g : F) : Int :=
  s b f - s b g - s a f + s a g

/-- Vanishing of every double exchange is equivalent to an additive log-score
surface. A single nonzero cycle rules out additivity on the tested grid. -/
theorem separable_iff {B F : Type} (s : B → F → Int) (base : B) (field : F) :
    (∀ b f, cycle s b base f field = 0) ↔
    ∃ (u : B → Int) (v : F → Int), ∀ b f, s b f = u b + v f := by
  constructor
  · intro zero
    refine ⟨fun b => s b field, fun f => s base f - s base field, ?_⟩
    intro b f
    have h := zero b f
    unfold cycle at h
    change s b f = s b field + (s base f - s base field)
    omega
  · rintro ⟨u, v, h⟩ b f
    simp only [cycle,h]
    omega

/-- Averaging marginal changes across the two possible exchange orders
assigns precisely the total log-score gain; the factor two keeps integer units. -/
theorem attribution (both baseOnly fieldOnly neither : Int) :
    ((both-fieldOnly)+(baseOnly-neither)) + ((both-baseOnly)+(fieldOnly-neither)) =
    2*(both-neither) := by omega

#print axioms separable_iff
#print axioms attribution
end VisualSourcePaths
