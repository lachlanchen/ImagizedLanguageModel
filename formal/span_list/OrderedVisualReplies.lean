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

/-- Count pages satisfying an empirical image-level check. -/
def countPass (test : α → Bool) : List α → Nat
  | [] => 0
  | x :: xs => (test x).toNat + countPass test xs

/-- If two roles are written by a factored predictor, both are correct on at
least A + B - N requests. This distribution-free statement also applies when
the two component errors are correlated. Agreement of the joined sentence
with the source is an additional empirical condition. -/
theorem joint_coverage (a b : α → Bool) (xs : List α) :
    countPass a xs + countPass b xs ≤ xs.length + countPass (fun x => a x && b x) xs := by
  induction xs with
  | nil => simp [countPass]
  | cons x xs ih =>
    simp only [countPass, List.length_cons]
    cases ha : a x <;> cases hb : b x <;> simp [ha,hb] <;> omega

/-- Expand a single valid insertion mark with original source cells. -/
def splice (left ink right : List α) : List α := left ++ (ink ++ right)

theorem drop_context (left ink right : List α) :
    (splice left ink right).drop left.length = ink ++ right := by
  induction left with
  | nil => rfl
  | cons _ _ ih => exact ih

theorem anchor_length (left ink right : List α) :
    (splice left ink right).length = left.length + ink.length + right.length := by
  simp [splice, Nat.add_assoc]

#print axioms drop_context
#print axioms joint_coverage
#print axioms order_blind_bound
#print axioms both_require_change
end OrderedVisualReplies
