import Std

namespace CoarseSpanTeacher

/-- An unnormalised mass for each of three source intervals. The first interval
is preserved individually, while the two others share a residual bucket. -/
def observed (m : Nat × Nat × Nat) := (m.1, m.2.1 + m.2.2)

theorem regroup (a b c : Nat) : a + (observed (a,b,c)).2 = (a+b)+c := by
  simp only [observed]
  omega

/-- Redistribution inside the residual bucket preserves every observable mass.
Consequently every loss which sees only these masses takes the same value. -/
theorem tail_unidentified {α : Type} (loss : Nat × Nat → α)
    (kept x y x' y' : Nat) (h : x + y = x' + y') :
    loss (observed (kept,x,y)) = loss (observed (kept,x',y')) := by
  simp only [observed,h]

/-- A top-ranked retained atom does not imply that all residual mass is small. -/
example : (6:Nat) > 5 ∧ (6:Nat) > 4 ∧ (observed (6,5,4)).2 > 6 := by decide
/-- An arbitrary interval list with masses in common fixed-point units. -/
def total : List Nat → Nat
  | [] => 0
  | x :: xs => x + total xs

theorem total_append (xs ys : List Nat) :
    total (xs ++ ys) = total xs + total ys := by
  induction xs with
  | nil => simp [total]
  | cons a xs ih => simp [total, ih, Nat.add_assoc]

def coarsen (kept dropped : List Nat) := kept ++ [total dropped]

theorem coarsen_preserves_mass (kept dropped : List Nat) :
    total (coarsen kept dropped) = total (kept ++ dropped) := by
  simp [coarsen, total_append, total]

theorem any_tail_unidentified {α : Type} (loss : List Nat → α)
    (kept xs ys : List Nat) (h : total xs = total ys) :
    loss (coarsen kept xs) = loss (coarsen kept ys) := by
  simp only [coarsen, h]

/-- If the teacher and student kept-bin masses agree, even exact agreement
on all observable bins permits opposing best intervals inside the tail. -/
example : coarsen [6] [5,1] = coarsen [6] [1,5] := by decide

#print axioms coarsen_preserves_mass
#print axioms any_tail_unidentified
#print axioms regroup
#print axioms tail_unidentified
end CoarseSpanTeacher
