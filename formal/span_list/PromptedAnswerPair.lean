import Std
namespace PromptedAnswerPair
/-- For two distinct strings on one fixed page, the second-question masses
of these disjoint stopped outputs sum to at most the total. A per-event bound
on the question intervention therefore also bounds paired correct mass. -/
theorem paired_mass (p₁a p₂a p₂b total delta : Int)
    (exclusive : p₂a + p₂b ≤ total) (effect : p₁a - p₂a ≤ delta) :
    p₁a + p₂b ≤ total + delta := by omega

theorem necessary_effect (p₁a p₂a p₂b total excess : Int)
    (exclusive : p₂a + p₂b ≤ total) (observed : total + excess ≤ p₁a + p₂b) :
    excess ≤ p₁a - p₂a := by omega

/-- Exact, deterministic writing on both questions with different targets
requires different emitted images for the two queries. -/
theorem paired_writing {α : Type} (a b y z : α) (different : a ≠ b)
    (first : y = a) (second : z = b) : y ≠ z := by
  intro h
  exact different (first.symm.trans (h.trans second))
end PromptedAnswerPair
