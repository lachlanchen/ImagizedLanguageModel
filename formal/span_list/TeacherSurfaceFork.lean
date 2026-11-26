import Std

namespace TeacherSurfaceFork

/-- Additive sequence score of a causal writer after a committed history.
    Integers permit exact verification at a specified log-score precision.
    The terminal stop is included as a symbol. -/
def writing {G : Type} (step : List G → G → Int) (hist : List G) : List G → Int
  | [] => 0
  | g :: tail => step hist g + writing step (hist ++ [g]) tail

theorem append_score {G : Type} (s : List G → G → Int) (history left right : List G) :
    writing s history (left ++ right) = writing s history left +
      writing s (history ++ left) right := by
  induction left generalizing history with
  | nil => simp [writing]
  | cons g rest ih =>
    change s history g + writing s (history ++ [g]) (rest ++ right) =
      (s history g + writing s (history ++ [g]) rest) + writing s (history ++ (g :: rest)) right
    rw [ih]
    simp [List.append_assoc, Int.add_assoc]

/-- For two attested spellings beginning with exactly the same ink, their
    complete score difference can be measured at the fork and in its tails.
    Prefix frequency cancels, including any preceding stop-free glyphs. -/
theorem fork_gap {G : Type} (s : List G → G → Int) (p u v : List G) (a b : G) :
    writing s [] (p ++ a :: u) - writing s [] (p ++ b :: v) =
    (s p a - s p b) + (writing s (p ++ [a]) u - writing s (p ++ [b]) v) := by
  rw [append_score, append_score]
  simp only [List.nil_append, writing]
  omega

/-- Subtract two writers' odds to assign the intervention effect exactly.
    A gain attributable to post-fork teacher forcing need not change the
    preference at the first distinct glyph. -/
theorem intervention {G : Type} (new old : List G → G → Int)
    (p u v : List G) (a b : G) :
    (writing new [] (p ++ a :: u) - writing new [] (p ++ b :: v)) -
    (writing old [] (p ++ a :: u) - writing old [] (p ++ b :: v)) =
    ((new p a - new p b) - (old p a - old p b)) +
    ((writing new (p ++ [a]) u - writing new (p ++ [b]) v) -
      (writing old (p ++ [a]) u - writing old (p ++ [b]) v)) := by
  rw [fork_gap, fork_gap]
  omega

end TeacherSurfaceFork
