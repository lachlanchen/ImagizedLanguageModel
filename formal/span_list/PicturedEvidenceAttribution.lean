import Std

namespace PicturedEvidenceAttribution

/-- Each corner is the difference between two complete-answer scores, using
one common fixed-point unit. The first bit restores the real source image;
the second restores the phrase draft. Double the contributions to avoid
rounding any half unit in the averaged restoration paths. -/
def source2 (empty source draft both : Int) := (source-empty) + (both-draft)
def draft2 (empty source draft both : Int) := (draft-empty) + (both-source)

theorem account (a b c d : Int) :
    source2 a b c d + draft2 a b c d = 2*(d-a) := by
  unfold source2 draft2
  omega

/-- Efficiency together with equal division of the interaction beyond
either channel's isolated effect uniquely specifies both contributions. -/
theorem unique_split (a b c d x y : Int)
    (total : x+y=2*(d-a)) (interaction : x-2*(b-a)=y-2*(c-a)) :
    x=source2 a b c d ∧ y=draft2 a b c d := by
  unfold source2 draft2
  omega

/-- The intact reference outranks the free answer precisely when the
two channels together pay the initial deficit. Either channel may help
while the final answer still has the higher score. -/
theorem reverses (a b c d : Int) :
    0 < d ↔ -2*a < source2 a b c d + draft2 a b c d := by
  rw [account]
  omega

end PicturedEvidenceAttribution
