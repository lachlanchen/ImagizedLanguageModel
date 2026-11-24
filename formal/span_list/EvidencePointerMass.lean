import Std

namespace EvidencePointerMass
/-- a and b are the correct and incorrect mass inside the evidence interval.
c and d are the corresponding masses outside. Cross multiplication avoids
rounding and compares the normalized probability of the *word*, including
every matching position. An integer boost s+2 is strictly larger than one. -/
theorem boostIff (a b c d s : Nat) :
    (a+c)*((s+2)*(a+b)+c+d) < ((s+2)*a+c)*(a+b+c+d) ↔ b*c < a*d := by
  have positive : 0 < s+1 := Nat.succ_pos _
  have reduce : (s+1)*(b*c) < (s+1)*(a*d) ↔ b*c < a*d :=
    Nat.mul_lt_mul_left positive
  rw [← reduce]
  simp only [Nat.add_mul, Nat.mul_add, Nat.one_mul, Nat.two_mul, Nat.mul_one,
    Nat.mul_two, Nat.mul_comm, Nat.mul_left_comm, Nat.mul_assoc]
  omega

/-- If the evidence contains no correct candidate but discounts a nonzero
correct candidate outside, boosting cannot improve its probability. -/
theorem noCorrectInside (b c d s : Nat) :
    ¬ ((0+c)*((s+2)*(0+b)+c+d) < ((s+2)*0+c)*(0+b+c+d)) := by
  rw [boostIff]
  simp

#print axioms boostIff
#print axioms noCorrectInside
end EvidencePointerMass
