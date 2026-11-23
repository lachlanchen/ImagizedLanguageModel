import Std

namespace BilingualReading

/-- A complete page asks for two incompatible facts in each of two
languages. The reply may have alternative accepted surface forms. -/
def complete {Reply : Type} (first second : Reply → Prop)
    (answer : Bool → Bool → Reply) : Prop :=
  ∀ lang, first (answer lang false) ∧ second (answer lang true)

theorem complete_distinguishes_intent {Reply : Type}
    (first second : Reply → Prop) (incompatible : ∀ r, first r → ¬ second r)
    (answer : Bool → Bool → Reply) (ok : complete first second answer) :
    ∀ lang, answer lang false ≠ answer lang true := by
  intro lang eq
  have h := ok lang
  exact incompatible _ (eq ▸ h.1) h.2

/-- On a fixed page any deterministic encoder that collapses the two
requests in one language forces a downstream failure, independently
of what the language writer does in the other language. -/
theorem collision_prevents_complete {Ink Feature Reply : Type}
    (q : Bool → Bool → Ink) (encode : Ink → Feature) (write : Feature → Reply)
    (a b : Reply → Prop) (disjoint : ∀ r, a r → ¬ b r)
    (lang : Bool) (collision : encode (q lang false) = encode (q lang true)) :
    ¬ complete a b (fun l intent => write (encode (q l intent))) := by
  intro ok
  exact (complete_distinguishes_intent a b disjoint _ ok lang) (congrArg write collision)

#print axioms complete_distinguishes_intent
#print axioms collision_prevents_complete
end BilingualReading
