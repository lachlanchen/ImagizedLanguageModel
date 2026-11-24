import Std

namespace EvidenceViewTransfer

/-- Both views use the same probability denominator. At a reached target prefix,
a supplied-evidence advantage can certify the next full-page decision given a
bound on the loss of target mass. Rival may be any other glyph or stop. -/
theorem full_wins (total crop full rival drift : Int)
    (disjoint : full + rival ≤ total) (drop : crop - full ≤ drift)
    (margin : total + 2 * drift < 2 * crop) : rival < full := by omega

/-- Each wrong or tied full-page choice requires either insufficient cropped
confidence or a drop larger than the allocated tolerance, pointwise. -/
theorem account_mistake (total crop full rival drift : Int)
    (disjoint : full + rival ≤ total) (mistake : full ≤ rival) :
    2 * crop ≤ total + 2 * drift ∨ drift < crop - full := by omega

end EvidenceViewTransfer
