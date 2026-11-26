import Std
namespace OrderedStripLikelihood

/-- Scaled two-position likelihood for two target strips, with a uniform start.
The four emission products and the four transition masses may share any fixed
denominators. Comparing the target orders cancels diagonal path contributions. -/
def mass (p q r s x y z w : Int) : Int := p*x + q*y + r*z + s*w

theorem order_difference (p q r s x y z w : Int) :
    mass p q r s x y z w - mass p q r s x z y w = (q-r)*(y-z) := by
  simp only [mass, Int.sub_mul, Int.mul_sub]
  omega

/-- The direction of the learned move and the direction of the two lexical
cross-products suffice to distinguish a target order. -/
theorem prefers_first (p q r s x y z w : Int) (move : r<q) (lex : z<y) :
    mass p q r s x z y w < mass p q r s x y z w := by
  have delta := order_difference p q r s x y z w
  have pos : 0 < (q-r)*(y-z) := Int.mul_pos (by omega) (by omega)
  omega

end OrderedStripLikelihood
