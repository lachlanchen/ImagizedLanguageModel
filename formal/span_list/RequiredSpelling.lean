import Std

namespace RequiredSpelling

/-- Each visible unit may be an entire word image after an allowed edge edit.
The predicates include exactly the words available to the two writing routes. -/
inductive Writes {W : Type} (visible writable : W → Bool) : List W → Nat → Prop
  | stop : Writes visible writable [] 0
  | copy (hx : visible x = true) (tail : Writes visible writable xs k) :
      Writes visible writable (x :: xs) k
  | spell (hx : writable x = true) (tail : Writes visible writable xs k) :
      Writes visible writable (x :: xs) (k+1)

def required {W : Type} (visible : W → Bool) : List W → Nat
  | [] => 0
  | x :: xs => (if visible x then 0 else 1) + required visible xs

/-- Even repeated occurrences of an absent word require separate spelling actions. -/
theorem lower {W : Type} {visible writable : W → Bool} {xs : List W} {k : Nat}
    (path : Writes visible writable xs k) : required visible xs ≤ k := by
  induction path with
  | stop => simp [required]
  | copy hx _ ih => simpa [required,hx] using ih
  | @spell x _ _ _ _ ih =>
    cases h : visible x <;> simp [required,h] <;> omega

/-- The lower bound is constructive when all absent target words are writable.
A decoder can follow the reference in written order and stop. -/
theorem attained {W : Type} (visible writable : W → Bool) (xs : List W)
    (coverage : ∀ x ∈ xs, visible x = false → writable x = true) :
    Writes visible writable xs (required visible xs) := by
  induction xs with
  | nil => exact .stop
  | cons x xs ih =>
    have tail := ih (fun y mem => coverage y (List.mem_cons_of_mem _ mem))
    cases h : visible x with
    | true => simpa [required,h] using Writes.copy h tail
    | false =>
      have hw := coverage x (by simp) h
      simpa [required,h,Nat.add_comm] using Writes.spell hw tail

/-- The paired assignment cancels a fixed bias for each question and answer.
Integer scores allow exact arithmetic, including fixed-point log scores. -/
theorem paired_assignment_invariant (a b c d q₁ q₂ r₁ r₂ : Int) :
    (a + q₁ + r₁) + (d + q₂ + r₂) > (b + q₁ + r₂) + (c + q₂ + r₁) ↔
      a + d > b + c := by omega

end RequiredSpelling
