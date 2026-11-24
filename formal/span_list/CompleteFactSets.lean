import Std
namespace CompleteFactSets

def total : List Int → Int
  | [] => 0
  | a :: xs => a + total xs

/-- Sorted m-weakest deletion and any other m-deletion share some runs.
Pair the remaining runs arbitrarily. A threshold separating the weakest
from the others orders each pair, even for negative gain. -/
theorem ordered_pair_sum (xs : List (Int × Int)) (t : Int)
    (h : ∀ z ∈ xs, z.1 ≤ t ∧ t ≤ z.2) :
    total (xs.map Prod.fst) ≤ total (xs.map Prod.snd) := by
  induction xs with
  | nil => simp [total]
  | cons z zs ih =>
    have a := h z (by simp)
    have b := ih (fun w hw => h w (by simp [hw]))
    simp only [List.map_cons, total]
    omega

/-- After cancelling the shared deleted runs, deleting the weakest m
maximizes the remaining score among all deletions of that cardinality.
The terminal reward may depend arbitrarily on the remaining count. -/
theorem strongest_omission (full common reward t : Int) (xs : List (Int × Int))
    (h : ∀ z ∈ xs, z.1 ≤ t ∧ t ≤ z.2) :
    full - (common + total (xs.map Prod.snd)) + reward ≤
    full - (common + total (xs.map Prod.fst)) + reward := by
  have := ordered_pair_sum xs t h
  omega

/-- Moving one non-overlapping fact changes exactly its local gain; deleting
or adding also changes the final reward. Neighbour costs are included in
the unary potentials by InkRunSurvival.shift_exit. -/
theorem score_changes (rest gold rival ck smaller larger : Int) :
    (rest + gold + ck) - (rest + rival + ck) = gold - rival ∧
    (rest + gold + ck) - (rest + smaller) = gold + ck - smaller ∧
    (rest + gold + ck) - (rest + gold + rival + larger) = ck - larger - rival := by
  omega

#print axioms strongest_omission
#print axioms score_changes
end CompleteFactSets
