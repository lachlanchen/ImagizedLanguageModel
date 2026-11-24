import Std

namespace OpticalPhrasePath

/-- At each reference boundary all origins emitting the selected ink run can be
combined, because the recurrent state depends only on the emitted pixels.
In common nonnegative mass units, alternatives represents other correct
segmentations, scaled to the same denominator at this boundary. -/
structure Boundary where
  origins : List Nat
  alternatives : Nat

def chosen : List Boundary → Nat
  | [] => 1
  | x :: xs => (x.origins.foldl (· + ·) 0) * chosen xs

def stoppedMass : List Boundary → Nat
  | [] => 1
  | x :: xs => (x.origins.foldl (· + ·) 0) * stoppedMass xs + x.alternatives

def omitted : List Boundary → Nat
  | [] => 0
  | x :: xs => (x.origins.foldl (· + ·) 0) * omitted xs + x.alternatives

/-- Exact slack of the supervised segmentation lower bound, including the
final stop as an action. No assumption of independence across glyphs. -/
theorem path_decomposition (xs : List Boundary) :
    stoppedMass xs = chosen xs + omitted xs := by
  induction xs with
  | nil => rfl
  | cons x xs ih =>
    simp only [stoppedMass, chosen, omitted, ih, Nat.mul_add]
    omega

theorem path_lower_bound (xs : List Boundary) : chosen xs ≤ stoppedMass xs := by
  rw [path_decomposition]
  omega

/-- Replacing a stretch of single-cell decisions by one run improves that
path's mass exactly when the run beats their product, for a shared positive
prefix and suffix. The two middle masses must first be scaled to the same
denominator. -/
theorem span_gain (prefixMass suffixMass singleMass runMass : Nat)
    (hp : 0 < prefixMass) (hs : 0 < suffixMass) :
    prefixMass * singleMass * suffixMass < prefixMass * runMass * suffixMass ↔ singleMass < runMass := by
  rw [Nat.mul_lt_mul_right hs, Nat.mul_lt_mul_left hp]

end OpticalPhrasePath
