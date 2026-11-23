import Std

namespace FixedOrderRoles

/-- An answer is one of two distinct source images. The question asks for one
of two actions. Exchanging the actions leaves both names at their original
locations but reverses which image answers the same question. -/
def requested {α : Type} (left right : α) (question exchanged : Bool) :=
  if question != exchanged then right else left

theorem edit_changes_target {α : Type} (left right : α) (different : left ≠ right)
    (q : Bool) : requested left right q false ≠ requested left right q true := by
  cases q
  · exact different
  · exact Ne.symm different

/-- Even an arbitrary learned position preference that depends on the question
cannot be correct on both worlds if its reply ignores this page edit. -/
theorem invariant_reader_misses {α : Type} (left right : α) (different : left ≠ right)
    (reply : Bool → α) (q : Bool) :
    ¬ (reply q = requested left right q false ∧ reply q = requested left right q true) := by
  intro h
  exact edit_changes_target left right different q (h.1.symm.trans h.2)

/-- The complementary condition: a reader that identifies each action's
location can return the intended physical image in all four conditions. -/
theorem action_location_suffices {α : Type} (left right : α) (locate : Bool → Bool → Bool)
    (correct : ∀ q e, locate q e = (q != e)) (q e : Bool) :
    (if locate q e then right else left) = requested left right q e := by
  rw [correct]
  rfl

#print axioms invariant_reader_misses
#print axioms action_location_suffices
end FixedOrderRoles
