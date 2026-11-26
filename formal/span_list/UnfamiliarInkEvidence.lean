import Std

namespace UnfamiliarInkEvidence

/-- Two competing complete readings in each class of physically distinguishable inputs.
Other source cells, positions, draft and question are held fixed within a group. -/
abbrev Counts := Nat × Nat

def visibleOpt : List Counts → Nat
  | [] => 0
  | (a,b)::rest => max a b + visibleOpt rest

def countA : List Counts → Nat
  | [] => 0
  | (a,_)::rest => a + countA rest

def countB : List Counts → Nat
  | [] => 0
  | (_,b)::rest => b + countB rest

theorem eachConstantBound (xs : List Counts) :
    countA xs ≤ visibleOpt xs ∧ countB xs ≤ visibleOpt xs := by
  induction xs with
  | nil => simp [countA, countB, visibleOpt]
  | cons x xs ih =>
    rcases x with ⟨a,b⟩
    simp only [countA,countB,visibleOpt]
    have ha := Nat.le_max_left a b
    have hb := Nat.le_max_right a b
    omega

/-- Erasing all unfamiliar shapes collapses their decisions to one modal answer.
The difference is an opportunity for learning on the sampled distribution. -/
theorem erasureBound (xs : List Counts) :
    max (countA xs) (countB xs) ≤ visibleOpt xs := by
  rcases eachConstantBound xs with ⟨ha,hb⟩
  omega

/-- With two shape classes favouring opposite readings, the exact attainable gain
is the smaller majority margin. Counts may be zero or tied. -/
theorem gainForOppositeReadings (a b c d : Nat) (left : b ≤ a) (right : c ≤ d) :
    max a b + max c d = max (a+c) (b+d) + min (a-b) (d-c) := by
  omega

/-- Visually distinct shapes that favour the same reading give zero optimal gain. -/
theorem noGainForAlignedMajorities (a b c d : Nat) (left : b ≤ a) (right : d ≤ c) :
    max a b + max c d = max (a+c) (b+d) := by
  omega

end UnfamiliarInkEvidence
