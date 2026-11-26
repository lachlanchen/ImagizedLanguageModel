import Std

namespace GlyphAttentionMeasure

/-- Terms use fixed-point attention mass and signed value coordinates.
A tuple stores the number of image positions, per-position mass, original
phrase mass and one value coordinate. One common denominator suffices for
all rational masses in a finite example. -/
structure Group where
  copies : Int
  perCell : Int
  original : Int
  value : Int

def sumBy (f : Group → Int) : List Group → Int
  | [] => 0
  | h::t => f h + sumBy f t

theorem conserve (g : List Group) (den : Int) (v : Group → Int)
    (weights : ∀ j ∈ g, j.copies * j.perCell = den * j.original) :
    sumBy (fun j => j.copies * j.perCell * v j) g =
      den * sumBy (fun j => j.original * v j) g := by
  induction g with
  | nil => simp [sumBy]
  | cons a g ih =>
    have fixed := weights a (by simp)
    have rest := ih (fun j hj => weights j (by simp [hj]))
    simp only [sumBy]
    rw [rest, fixed, Int.mul_add, Int.mul_assoc]

/-- Conservation of both numerator and denominator implies identical
normalised attention at every query where the original total mass is positive.
Cross multiplication avoids committing to a particular fixed-point resolution. -/
theorem same_pool (g : List Group) (den : Int)
    (weights : ∀ j ∈ g, j.copies * j.perCell = den * j.original) :
    sumBy (fun j => j.copies * j.perCell * j.value) g * sumBy (fun j => j.original * 1) g =
      sumBy (fun j => j.copies * j.perCell * 1) g * sumBy (fun j => j.original * j.value) g := by
  rw [conserve g den (fun j => j.value) weights, conserve g den (fun _ => 1) weights]
  rw [Int.mul_assoc, Int.mul_assoc]
  congr 1
  exact Int.mul_comm _ _

end GlyphAttentionMeasure
