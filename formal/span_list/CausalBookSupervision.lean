import Std

namespace CausalBookSupervision

/-- A state after reading a prefix is independent of every possible continuation.
A transition consumes a pictured word and its already observed following separator. -/
def prefixState (step : σ → α → σ) (init : σ) (n : Nat) (xs : List α) : σ :=
  (xs.take n).foldl step init

theorem noFuture (step : σ → α → σ) (init : σ) (past a b : List α) :
    prefixState step init past.length (past ++ a) =
    prefixState step init past.length (past ++ b) := by
  simp [prefixState]

/-- An exposure is a (window, position) pair. One interval per pass covers all
eligible positions exactly once even when a book-window length is not divisible by four. -/
def quarter (n k : Nat) := (n*k)/4

theorem quartersTelescope (n : Nat) :
    (quarter n 1 - quarter n 0) + (quarter n 2 - quarter n 1) +
    (quarter n 3 - quarter n 2) + (quarter n 4 - quarter n 3) = n := by
  simp [quarter]
  omega

/-- For two labelled options in a mirrored pair, the same option-independent
context correction can be subtracted from each matched option across both
premises without changing the signed interaction. It cannot manufacture a
positive interaction for a reader that ignores the premise. -/
theorem mirrorInteraction (p₁ p₂ m₁ m₂ n₁ n₂ : Int) :
    ((p₁-n₁)-(p₂-n₂)) + ((m₂-n₂)-(m₁-n₁)) =
    (p₁-p₂) + (m₂-m₁) := by omega

theorem bothCorrectRequiresInteraction (p₁ p₂ m₁ m₂ : Int)
    (one : p₁ > p₂) (two : m₂ > m₁) : (p₁-p₂)+(m₂-m₁) > 0 := by omega

end CausalBookSupervision
