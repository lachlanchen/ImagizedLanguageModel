import Std

namespace RetrievedGlyphVote

/-- Scale the neural probabilities by a common positive integer unit.
A vote with strength a/b and n observed cells mixes with the neural
distribution at a*n/(2*b+b*n). Dropping the common positive denominator
leaves the score below, including the n=0 abstention case. -/
def score (a b n unit neural count : Int) : Int :=
  (2*b+n*(b-a))*neural + a*unit*count

/-- A supported glyph overtakes the neural favourite precisely when its
weighted count advantage exceeds the scaled neural probability deficit. -/
theorem overtake_iff (a b n unit p q c d : Int) :
    score a b n unit p c > score a b n unit q d ↔
    (a*unit)*(c-d) > (2*b+n*(b-a))*(q-p) := by
  unfold score
  simp only [Int.mul_sub]
  omega

theorem no_evidence_keeps_order (a b unit p q : Int) (hb : 0 < b) :
    score a b 0 unit p 0 > score a b 0 unit q 0 ↔ p > q := by
  simp only [score, Int.zero_mul, Int.add_zero, Int.mul_zero]
  have pos : 0 < 2*b := by omega
  constructor
  · intro h
    exact Int.lt_of_mul_lt_mul_left h (by omega)
  · intro h
    exact Int.mul_lt_mul_of_pos_left h pos

end RetrievedGlyphVote
