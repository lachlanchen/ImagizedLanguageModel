import Std

namespace StoppedSentenceSearch

/-- A score may be a probability, a log probability, or another ordered
quantity. Appending any glyph, including STOP, must never raise it. -/
theorem suffix_bound {G S : Type} (le : S → S → Prop)
    (refl : ∀ x, le x x) (trans : ∀ {x y z}, le x y → le y z → le x z)
    (score : List G → S)
    (step : ∀ p a, le (score (p ++ [a])) (score p))
    (pre tail : List G) : le (score (pre ++ tail)) (score pre) := by
  induction tail generalizing pre with
  | nil => simpa using refl (score pre)
  | cons a rest ih =>
    have h := trans (ih (pre ++ [a])) (step pre a)
    simpa [List.append_assoc] using h

/-- If every legal stopped stream was examined or extends a discarded
prefix, the best examined completion is globally maximal whenever it
dominates every discarded prefix. Applies to a finite decision budget. -/
theorem certified_mode {G S : Type} (le : S → S → Prop)
    (refl : ∀ x, le x x) (trans : ∀ {x y z}, le x y → le y z → le x z)
    (score : List G → S) (step : ∀ p a, le (score (p ++ [a])) (score p))
    (legal seen discarded : List G → Prop) (best : List G)
    (partition : ∀ y, legal y → seen y ∨ ∃ p s, discarded p ∧ y = p ++ s)
    (best_seen : ∀ y, seen y → le (score y) (score best))
    (cut : ∀ p, discarded p → le (score p) (score best)) :
    ∀ y, legal y → le (score y) (score best) := by
  intro y hy
  rcases partition y hy with h | ⟨p, s, hp, rfl⟩
  · exact best_seen y h
  · exact trans (suffix_bound le refl trans score step p s) (cut p hp)

#print axioms certified_mode
end StoppedSentenceSearch
