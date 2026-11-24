import Std

namespace UnspecifiedReferent

/-- A single underspecified tag supplies the same supervision for two
referents. A decoder using only that tag cannot identify both readings. -/
theorem need_referent {Tag Word : Type} (decode : Tag → Word)
    (tag : Tag) (first second : Word) (distinct : first ≠ second) :
    ¬ (decode tag = first ∧ decode tag = second) := by
  rintro ⟨ha, hb⟩
  exact distinct (ha.symm.trans hb)

#print axioms need_referent
end UnspecifiedReferent
