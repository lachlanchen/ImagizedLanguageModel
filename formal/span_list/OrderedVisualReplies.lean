import Std

namespace OrderedVisualReplies

/-- The evaluator recovers two distinct fact positions from an emitted bitmap
stream. An absent or ambiguous occurrence receives no position. -/
def forward (positions : Option (Nat × Nat)) : Bool :=
  match positions with
  | some (a, b) => decide (a < b)
  | none => false

def backward (positions : Option (Nat × Nat)) : Bool :=
  match positions with
  | some (a, b) => decide (b < a)
  | none => false

/-- A single physical reply cannot receive both order credits on a page. -/
theorem incompatible (positions : Option (Nat × Nat)) :
    ¬ (forward positions = true ∧ backward positions = true) := by
  cases positions with
  | none => simp [forward]
  | some pair =>
    rcases pair with ⟨a, b⟩
    simp [forward, backward]
    omega

def credits : List (Option (Nat × Nat)) → Nat
  | [] => 0
  | p :: ps => (forward p).toNat + (backward p).toNat + credits ps

/-- Upper bound for a predictor which ignores request order on every page.
Stopping or legibility requirements can only remove additional credits. -/
theorem order_blind_bound (xs : List (Option (Nat × Nat))) :
    credits xs ≤ xs.length := by
  induction xs with
  | nil => decide
  | cons p ps ih =>
    have h := incompatible p
    have one : (forward p).toNat + (backward p).toNat ≤ 1 := by
      cases a : forward p <;> cases b : backward p <;> simp_all
    simp only [credits, List.length_cons]
    omega

/-- An order-sensitive writer must change its physical answer to earn both credits. -/
theorem both_require_change (read : α → Option (Nat × Nat)) (x y : α)
    (a : forward (read x) = true) (b : backward (read y) = true) : x ≠ y := by
  intro same
  subst y
  exact incompatible (read x) ⟨a,b⟩

#print axioms order_blind_bound
#print axioms both_require_change
end OrderedVisualReplies
