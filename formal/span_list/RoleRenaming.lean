import Std

namespace RoleRenaming

/-- Common-scale sum of supported-minus-rival log scores, measured in the
original name assignment and after consistently relabelling all occurrences. -/
def pooled (original renamed : Int) := original + renamed

theorem invariant (a b : Int) : pooled a b = pooled b a := by
  unfold pooled
  omega

theorem fixed_name_bias_cancels (a b bias : Int) :
    pooled (a + bias) (b - bias) = pooled a b := by
  unfold pooled
  omega

/-- If the same incorrect grounded preference persists across both views,
enforcing symmetry cannot make it correct. -/
theorem two_errors_remain (a b : Int) (ha : a ≤ 0) (hb : b ≤ 0) :
    pooled a b ≤ 0 := by
  unfold pooled
  omega

#print axioms fixed_name_bias_cancels
#print axioms two_errors_remain
end RoleRenaming
