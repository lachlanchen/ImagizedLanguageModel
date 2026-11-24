import Std

namespace OpticalRouteMargin

/-- Numerator of the score for a spelling gate g/d, in common integer mass units. -/
def mass (d g s c : Int) := g * s + (d-g) * c

/-- A pairwise comparison is exactly one linear constraint in the gate.
Intersect these constraints over all rivals to test whether rerouting could
make a particular glyph win at a fixed physical input and emitted prefix. -/
theorem halfspace (d g sa ca sb cb : Int) :
    mass d g sb cb < mass d g sa ca ↔
    d * (cb-ca) < g * ((sa-sb)-(ca-cb)) := by
  simp only [mass, Int.sub_mul, Int.mul_sub]
  omega

theorem weak_halfspace (d g sa ca sb cb : Int) :
    mass d g sb cb ≤ mass d g sa ca ↔
    d * (cb-ca) ≤ g * ((sa-sb)-(ca-cb)) := by
  simp only [mass, Int.sub_mul, Int.mul_sub]
  omega

theorem dominated (d g sa ca sb cb : Int) (h₀ : 0 ≤ g) (h₁ : g ≤ d)
    (hs : sa ≤ sb) (hc : ca ≤ cb) : mass d g sa ca ≤ mass d g sb cb := by
  have a := Int.mul_le_mul_of_nonneg_left hs h₀
  have b := Int.mul_le_mul_of_nonneg_left hc (show 0 ≤ d-g by omega)
  unfold mass
  omega

end OpticalRouteMargin
