import Std

namespace QuestionEvidence

/-- At fixed contextual states, a correction shared by the two intervals
cancels. Each interval correction includes its start and end adjustments. -/
theorem flips_iff (gold rival cg cr : Int) :
    rival + cr < gold + cg ↔ rival - gold < cg - cr := by omega

/-- A bound on each of the four affected endpoint scores certifies that some
baseline decisions cannot be reversed by the residual. -/
theorem preserve_interval (g r gs ge rs re b : Int)
    (gap : 4*b < g-r)
    (a : -b ≤ gs) (c : -b ≤ ge) (d : rs ≤ b) (e : re ≤ b) :
    r+rs+re < g+gs+ge := by omega

#print axioms flips_iff
#print axioms preserve_interval
end QuestionEvidence
