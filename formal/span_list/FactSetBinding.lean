import Std
namespace FactSetBinding
/-- u and v contain the count reward and the sums for the unchanged parts of
each answer on the same pictured page. x and y are disjoint run gains. -/
theorem exchange_identity (u v x y : Int) :
    (u+x) + (v+y) = (u+y) + (v+x) := by omega

/-- Even with question-conditioned scores, one of the two annotated lists
cannot strictly beat its exchanged rival when both exchanges are admissible. -/
theorem cannot_prefer_both (u v x y : Int) :
    ¬ (u+x > u+y ∧ v+y > v+x) := by omega

/-- A reward for two jointly selected runs can lift both desired lists.
a and b must be positive and larger than the opposing local imbalance. -/
theorem pair_reward (u v x y a b : Int) (h : a > y-x) (k : b > x-y) :
    u+x+a > u+y ∧ v+y+b > v+x := by omega
#print axioms cannot_prefer_both
end FactSetBinding

namespace FactSetBinding
theorem two_readings (base signed : Int) :
    max (base+signed) (base-signed) = base + max signed (-signed) := by omega

/-- With signed gains (+1,+1) and (-1,-1), both coherent pairs beat either exchange. -/
example : max (1+1 : Int) (-(1+1)) > max (1-1) (-(1-1)) ∧
    max (-1-1 : Int) (-(-1-1)) > max (-1+1) (-(-1+1)) := by decide
end FactSetBinding
