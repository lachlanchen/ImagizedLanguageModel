import Std

namespace ParagraphMargin

/-- Quantized scores for the annotated and competing paragraph. -/
structure Pair where
  correct : Int
  rival : Int

/-- Conservative ties count as an error. A stable first-index tie break can only
reduce the error count relative to this convention. -/
def miss (p : Pair) : Int := if p.correct ≤ p.rival then 1 else 0

/-- One score unit of separation after quantization. -/
def hinge (p : Pair) : Int := max 0 (1 + p.rival - p.correct)

theorem miss_le_hinge (p : Pair) : miss p ≤ hinge p := by
  unfold miss hinge
  split <;> omega

def total (f : Pair → Int) : List Pair → Int
  | [] => 0
  | p :: ps => f p + total f ps

theorem empirical_bound (ps : List Pair) :
    total miss ps ≤ total hinge ps := by
  induction ps with
  | nil => decide
  | cons p ps ih =>
    simp only [total]
    have h := miss_le_hinge p
    omega

/-- Correct order is stable if the true score gap is strictly greater than
twice the bound on each score perturbation (in the same integer units). -/
theorem stable_selection (a b a' b' ε : Int)
    (gap : 2 * ε < a - b) (lo : a - ε ≤ a') (hi : b' ≤ b + ε) :
    b' < a' := by omega

#print axioms empirical_bound
#print axioms stable_selection
end ParagraphMargin
