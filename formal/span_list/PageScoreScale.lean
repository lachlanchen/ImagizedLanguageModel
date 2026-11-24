import Std
namespace PageScoreScale

/-- Scaling every legal full-path score leaves its ordering intact. A gap
quoted in absolute model units therefore cannot alone certify better reading. -/
theorem order_and_gap (a b scale : Int) (positive : 0 < scale) (strict : a < b) :
    scale*a < scale*b ∧ scale*b - scale*a = scale*(b-a) := by
  exact ⟨Int.mul_lt_mul_of_pos_left strict positive, (Int.mul_sub _ _ _).symm⟩

/-- For an unrestricted maximum to reproduce a reference, the optimal
reference must at least tie every fixed legal rival in the same scoring system. -/
theorem escaped_old_rival (free gold old : Int) (old_le_free : old ≤ free)
    (gold_wins : gold = free) : old ≤ gold := by omega
end PageScoreScale
