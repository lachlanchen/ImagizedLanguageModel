import Std

namespace PrintRecovery

/-- The original answer's credit and whether the variant reproduced that answer
including its stop. Each row refers to the same source meaning and judge. -/
structure Case where
  original : Bool
  reproduced : Bool

def topCredit (x : Case) : Bool := x.original || !x.reproduced

def score : List Bool → Nat
  | [] => 0
  | b :: bs => b.toNat + score bs

/-- Every unchanged reply gets the original grade. Grades on the remaining
rows are unconstrained. -/
inductive Admissible : List Case → List Bool → Prop
  | nil : Admissible [] []
  | add {x xs b bs} : (x.reproduced = true → b = x.original) →
      Admissible xs bs → Admissible (x :: xs) (b :: bs)

theorem bounded {xs bs} (h : Admissible xs bs) :
    score bs ≤ score (xs.map topCredit) := by
  induction h with
  | nil => exact Nat.le_refl _
  | @add x xs b bs rule _ ih =>
    have point : b.toNat ≤ (topCredit x).toNat := by
      rcases x with ⟨o,e⟩
      cases e <;> cases o <;> cases b <;> simp_all [topCredit]
    exact Nat.add_le_add point ih

theorem attainable (xs : List Case) : Admissible xs (xs.map topCredit) := by
  induction xs with
  | nil => exact .nil
  | cons x xs ih =>
    apply Admissible.add _ ih
    intro eq
    simp [topCredit, eq]

theorem sharp (xs : List Case) :
    Admissible xs (xs.map topCredit) ∧
    ∀ bs, Admissible xs bs → score bs ≤ score (xs.map topCredit) :=
  ⟨attainable xs, fun _ h => bounded h⟩

#print axioms sharp
end PrintRecovery
