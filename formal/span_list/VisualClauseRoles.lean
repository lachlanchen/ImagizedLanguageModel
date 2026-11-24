import Std

namespace VisualClauseRoles

/-- The two offsets of a grammatical relation in the visible passage. -/
structure Arc where
  verb : Nat
  object : Nat
  deriving DecidableEq

inductive Role | verb | object deriving DecidableEq

def locate (a : Arc) : Role → Nat
  | .verb => a.verb
  | .object => a.object

/-- With distinct endpoints, a correct pair of written-to positions identifies
both the arc orientation and the assignment of the two pictured questions,
up to the unavoidable simultaneous exchange of both names. -/
theorem paired_identification (a target : Arc) (qv qo : Role)
    (distinct : target.verb ≠ target.object)
    (hv : locate a qv = target.verb) (ho : locate a qo = target.object) :
    (qv = .verb ∧ qo = .object ∧ a = target) ∨
    (qv = .object ∧ qo = .verb ∧ a.verb = target.object ∧ a.object = target.verb) := by
  rcases a with ⟨v,o⟩
  rcases target with ⟨t,u⟩
  cases qv <;> cases qo <;> simp_all [locate]

/-- A predictor requiring the verb to lie left of the object cannot reproduce
the endpoints of an attested object-first relation. -/
theorem fronted_excludes_fixed_order (pred target : Arc)
    (always : pred.verb < pred.object) (front : target.object < target.verb) :
    pred ≠ target := by
  rintro rfl
  omega

/-- Collapsing a minimal question pair is fatal on every page with distinct
targets, regardless of the arc scorer or the amount of document training. -/
theorem collapsed_questions_fail (a t : Arc) (r : Role) (h : t.verb ≠ t.object) :
    ¬ (locate a r = t.verb ∧ locate a r = t.object) := by
  rintro ⟨hv, ho⟩
  exact h (hv.symm.trans ho)

#print axioms collapsed_questions_fail
#print axioms paired_identification
#print axioms fronted_excludes_fixed_order
end VisualClauseRoles
