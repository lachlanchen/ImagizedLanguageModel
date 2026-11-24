import Std

namespace VisualFieldAlignment

variable {S : Type} [OfNat S 0] [Add S] [Mul S]

def rightMass : List (S × S) → S
  | [] => 0
  | (_, r) :: xs => r + rightMass xs

/-- Enumerate all products of a left weight at u and a right weight at v > u. -/
def allPairs : List (S × S) → S
  | [] => 0
  | (l, _) :: xs => l * rightMass xs + allPairs xs

/-- A scanned prefix contributes to every subsequent right observation. -/
def scan (p : S) : List (S × S) → S
  | [] => 0
  | (l, r) :: xs => p * r + scan (p + l) xs

/-- Every strictly increasing latent alignment is counted exactly once.
The operations can be ordinary nonnegative weights, or addition and
multiplication transported into log space. Floating-point rounding is
checked separately against enumeration in the implementation. -/
theorem precedingMass
    (addA : ∀ a b c : S, (a + b) + c = a + (b + c))
    (rightZero : ∀ a : S, a * 0 = 0)
    (zeroPlus : ∀ a : S, 0 + a = a)
    (distL : ∀ a b c : S, a * (b + c) = a * b + a * c)
    (distR : ∀ a b c : S, (a + b) * c = a * c + b * c)
    (xs : List (S × S)) (p : S) :
    scan p xs = p * rightMass xs + allPairs xs := by
  induction xs generalizing p with
  | nil => simp [scan, rightMass, allPairs, rightZero, zeroPlus]
  | cons x xs ih =>
    rcases x with ⟨l,r⟩
    simp only [scan, rightMass, allPairs, ih, distL, distR, addA]

#print axioms precedingMass
end VisualFieldAlignment
