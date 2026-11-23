import Std

namespace TypefaceCopy

/-- The reader may use the same feature for two visibly different marks.
A copy action addresses the original page. -/
inductive Action (n : Nat) (Ink : Type) where
  | copy (position : Fin n)
  | write (glyph : Ink)

def render {n : Nat} {Ink : Type} (page : Fin n → Ink) : Action n Ink → Ink
  | .copy i => page i
  | .write glyph => glyph

theorem read_invariance {n : Nat} {Ink Feature Decision : Type}
    (encode : Ink → Feature) (decide : (Fin n → Feature) → Decision)
    (p q : Fin n → Ink) (same : ∀ i, encode (p i) = encode (q i)) :
    decide (encode ∘ p) = decide (encode ∘ q) := by
  have h : encode ∘ p = encode ∘ q := funext same
  rw [h]

theorem copy_preserves_difference {n : Nat} {Ink : Type}
    (p q : Fin n → Ink) (i : Fin n) (h : p i ≠ q i) :
    render p (.copy i) ≠ render q (.copy i) := h

/-- Combining proposal masses by physical image can change the winning
outcome when the copied typeface differs from the fixed typeface. -/
example : 3 + 3 > 4 ∧ 3 < 4 := by decide
end TypefaceCopy

namespace TypefaceCopy

/-- A lower bound on the winning score and a gap to every rival certify
the very same accept decision throughout a uniform score perturbation.
Scores, threshold and slack are expressed in common integer units. -/
theorem stable_acceptance {Label : Type} (s t : Label → Int)
    (winner : Label) (threshold margin eps : Int)
    (close_lo : ∀ i, s i - eps ≤ t i) (close_hi : ∀ i, t i ≤ s i + eps)
    (confidence : threshold + eps ≤ s winner)
    (separation : ∀ j, j ≠ winner → s j + margin + 2 * eps ≤ s winner) :
    threshold ≤ t winner ∧ ∀ j, j ≠ winner → t j + margin ≤ t winner := by
  constructor
  · have := close_lo winner
    omega
  · intro j different
    have := close_lo winner
    have := close_hi j
    have := separation j different
    omega

#print axioms stable_acceptance
end TypefaceCopy
