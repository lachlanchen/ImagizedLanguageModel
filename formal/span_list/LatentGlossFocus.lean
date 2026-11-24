import Std

namespace LatentGlossFocus

/-- With a fixed amount of probability mass elsewhere, moving a positive
    amount from the rival focus to the correct one improves the likelihood
    of the observed ink precisely when its emission is more likely there.
    The integers are counts with a common denominator. -/
theorem move_mass (elsewhere left right a b mass : Nat) (positive : 0 < mass) :
    (elsewhere + left*a + (right+mass)*b <
     elsewhere + (left+mass)*a + right*b) ↔ b < a := by
  simp only [Nat.add_mul]
  constructor
  · intro h
    by_cases hab : a ≤ b
    · have he := Nat.mul_le_mul_left mass hab
      omega
    · omega
  · intro h
    have he := Nat.mul_lt_mul_of_pos_left h positive
    omega

theorem flat_emission (left right mass emission : Nat) :
    left*emission + (right+mass)*emission =
    (left+mass)*emission + right*emission := by
  simp only [Nat.add_mul]
  omega

end LatentGlossFocus
