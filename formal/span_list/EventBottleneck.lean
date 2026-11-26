import Std

namespace EventBottleneck

/-- Errors of a stopped, deterministic image writer on one paired contrast.
An answer can contain both a physical bitmap and a stop flag. -/
def miss [DecidableEq A] (answer target : A) : Nat :=
  if answer = target then 0 else 1

/-- Once opposite required answers have identical source representations,
at least one error is unavoidable, irrespective of the writer weights. -/
theorem contrast_loss [DecidableEq A] (encode : S → R) (write : R → A)
    (p q : S) (a b : A) (same : encode p = encode q) (different : a ≠ b) :
    1 ≤ miss (write (encode p)) a + miss (write (encode q)) b := by
  rw [same]
  by_cases h : write (encode q) = a
  · simp [miss, h, different]
  · simp [miss, h]

def total : List Nat → Nat
  | [] => 0
  | x :: xs => x + total xs

/-- The bound adds over disjoint minimal pairs. The indicator counts only
observed collisions with distinct gold answers. -/
theorem panel_loss [DecidableEq R] [DecidableEq A] (encode : S → R) (write : R → A)
    (pairs : List (S × S × A × A)) :
    total (pairs.map (fun (p,q,a,b) => if encode p = encode q ∧ a ≠ b then 1 else 0)) ≤
    total (pairs.map (fun (p,q,a,b) => miss (write (encode p)) a + miss (write (encode q)) b)) := by
  induction pairs with
  | nil => simp [total]
  | cons item rest ih =>
    rcases item with ⟨p,q,a,b⟩
    simp only [List.map_cons, total]
    apply Nat.add_le_add _ ih
    split
    · rename_i h
      exact contrast_loss encode write p q a b h.1 h.2
    · exact Nat.zero_le _

end EventBottleneck
