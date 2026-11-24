import Std

namespace SentenceContext

/-- The gap is a feedback observation. It is never a required output glyph. -/
def collapsed {G : Type} (left right : List G) : List (Option G) :=
  left.map some ++ [none] ++ right.map some

/-- At every right-hand position the auxiliary teacher exposes exactly the
original left context, one gap, and the preceding right context. No character
of the bound value can leak through the previous-output channel. -/
theorem prefix_at_right {G : Type} (left right : List G) (k : Nat) :
    (collapsed left right).take (left.length + 1 + k) =
      left.map some ++ [none] ++ (right.take k).map some := by
  unfold collapsed
  have h : (left.map some ++ [none] : List (Option G)).length = left.length + 1 := by simp
  rw [← h, List.take_append, List.map_take]

/-- Include stop as a scored position. The supervision removed is exactly one
position for each actual cell of the value, however long it is. -/
theorem budget {G : Type} (left fact right : List G) :
    left.length + right.length + 1 + fact.length =
    (left ++ fact ++ right).length + 1 := by simp; omega

end SentenceContext
