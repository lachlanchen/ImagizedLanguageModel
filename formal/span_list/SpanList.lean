import Std

namespace SpanList
variable {G : Type} [DecidableEq G]

def weight (ref : G) : List (G × Nat) → Nat
  | [] => 0
  | (v,w) :: xs => (if v = ref then w else 0) + weight ref xs

theorem split_identical (ref v : G) (a b : Nat) (xs : List (G × Nat)) :
    weight ref ((v,a+b)::xs) = weight ref ((v,a)::(v,b)::xs) := by
  by_cases h : v = ref <;> simp [weight,h,Nat.add_assoc]

omit [DecidableEq G] in
theorem prefix_preserves_coverage (ref : G) (small large : List G)
    (nested : ∀ x, x ∈ small → x ∈ large) (found : ref ∈ small) : ref ∈ large :=
  nested ref found

theorem bigger_list_can_lose :
    weight 0 [(0,2)] > weight 1 [(0,2)] ∧
    weight 0 [(0,2),(1,3)] < weight 1 [(0,2),(1,3)] := by decide

#print axioms split_identical
#print axioms prefix_preserves_coverage
#print axioms bigger_list_can_lose
end SpanList

