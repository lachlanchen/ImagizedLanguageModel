import Std

namespace BookOrderPrior

def marginalCost (cost : α → Int) : List α → Int
  | [] => 0
  | x :: xs => cost x + marginalCost cost xs

/-- An order-blind glyph code gives each matched window exactly the same cost. -/
theorem permutationCost (cost : α → Int) (p : List.Perm xs ys) :
    marginalCost cost xs = marginalCost cost ys := by
  induction p with
  | nil => rfl
  | cons _ _ ih => simp [marginalCost, ih]
  | swap => simp [marginalCost]; omega
  | trans _ _ a b => exact a.trans b

/-- A word includes its first and last glyph and its accumulated internal edge score.
The following law accounts for the entire first-order path through adjacent words. -/
structure Word (α : Type) where
  first : α
  last : α
  inside : Int

def across (edge : α → α → Int) (l r : α) (u v : Word α) : Int :=
  edge l u.first + u.inside + edge u.last v.first + v.inside + edge v.last r

theorem exchangeEdges (edge : α → α → Int) (l r : α) (u v : Word α) :
    across edge l r u v - across edge l r v u =
    (edge l u.first - edge l v.first) + (edge u.last v.first - edge v.last u.first) +
    (edge v.last r - edge u.last r) := by
  unfold across
  omega

/-- Changing a source-independent prior cannot change the between-source log-odds
interaction at a shared target prefix. Scores are in integer fixed-point units. -/
theorem pairedInteraction (xa xb ya yb pa pb : Int) :
    ((xa+pa)-(xb+pb))-((ya+pa)-(yb+pb)) = (xa-xb)-(ya-yb) := by omega

/-- A shared bias can make both opposite choices strictly correct exactly when
their underlying score margins separate by at least two integer units. -/
theorem sharedBiasFeasible (x y : Int) :
    (∃ b : Int, x+b > 0 ∧ y+b < 0) ↔ x-y ≥ 2 := by
  constructor
  · rintro ⟨b,hx,hy⟩
    omega
  · intro h
    exact ⟨1-x, by omega, by omega⟩

end BookOrderPrior
