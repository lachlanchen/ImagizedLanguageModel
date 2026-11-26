import Std

namespace SourcePathRepair

/-- Each row contains the score of the writer given the inferred addresses,
    the score given the reference addresses, and whether their paths coincide. -/
structure Case where
  inferred : Bool
  supplied : Bool
  samePath : Bool

def count (f : Case → Bool) : List Case → Nat
  | [] => 0
  | r :: rs => (if f r then 1 else 0) + count f rs

/-- Resetting a deterministic writer to exactly the original source path
    can gain only the cases where supplied addresses repair the complete answer,
    and can also lose a previous success. -/
theorem repair_balance (rs : List Case) :
    count (·.supplied) rs + count (fun r => r.inferred && !r.supplied) rs =
    count (·.inferred) rs + count (fun r => r.supplied && !r.inferred) rs := by
  induction rs with
  | nil => rfl
  | cons r rs ih =>
    cases r with
    | mk i s e => cases i <;> cases s <;> simp [count] at * <;> omega

/-- Gives a lower bound on failures that survive even an exact source reader.
    Equal-path numerical disagreements can be counted and subtracted explicitly.
    Boolean scores require the same stopped-answer judgment for identical outputs. -/
theorem surviving_failures (rs : List Case) :
    count (fun r => r.samePath && !r.inferred) rs ≤
    count (fun r => !r.supplied) rs +
      count (fun r => r.samePath && (r.inferred != r.supplied)) rs := by
  induction rs with
  | nil => simp [count]
  | cons r rs ih =>
    cases r with
    | mk i s e => cases i <;> cases s <;> cases e <;> simp [count] at * <;> omega

#print axioms repair_balance
#print axioms surviving_failures

end SourcePathRepair
