import Std

namespace RoleLocationLearning

/-- w and v are unnormalised spatial masses at a common fixed precision.
A probability-conserving move of d units toward a window of higher role
occupancy increases matching mass by d times the change in occupancy.
Other windows and heads contribute c. Fractions share a common denominator. -/
theorem transfer (w v d low high c : Int) :
    ((w-d)*low + (v+d)*high + c) - (w*low + v*high + c)
     = d*(high-low) := by
  simp only [Int.sub_mul, Int.add_mul, Int.mul_sub]
  omega

theorem improvement (w v d low high c : Int)
    (moves : 0 < d) (better : low < high) :
    w*low + v*high + c < (w-d)*low + (v+d)*high + c := by
  have delta := transfer w v d low high c
  have pos := Int.mul_pos moves (show 0 < high-low by omega)
  omega

/-- Any exchange between equally labelled windows leaves the observed
role mass invariant, even when their represented words differ. Thus the
role loss by itself cannot identify which of those words was translated. -/
theorem equal_occupancy (w v d a c : Int) :
    (w-d)*a + (v+d)*a + c = w*a + v*a + c := by
  have h := transfer w v d a a c
  simp at h
  omega

end RoleLocationLearning
