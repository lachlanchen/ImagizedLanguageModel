import Std

namespace BoundedGlyphEvidence

/-- After applying the coefficient, take fixed-point log scores and bonus
    values at any common resolution. Every changed pair must satisfy this
    identity, including pairs with the terminal action. -/
theorem odds_change (a b u v : Int) :
    (a + u) - (b + v) = (a - b) + (u - v) := by omega

/-- A bounded cue cannot promote an arbitrarily improbable output. The bound
    applies against every competing glyph, in particular the original mode. -/
theorem maximum_regret (winner other bw bo bound : Int)
    (chosen : other + bo ≤ winner + bw)
    (upper : bw ≤ bound) (lower : -bound ≤ bo) :
    other - winner ≤ 2 * bound := by omega

theorem unchanged_strict_winner (a b u v bound : Int)
    (margin : 2 * bound < a - b)
    (low : -bound ≤ u) (high : v ≤ bound) :
    b + v < a + u := by omega

/-- A common shift in either comparison has no effect on preference. -/
theorem gauge (a b u v common : Int) :
    a + (u + common) ≥ b + (v + common) ↔ a + u ≥ b + v := by omega

end BoundedGlyphEvidence
