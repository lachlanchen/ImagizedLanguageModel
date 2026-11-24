import Std

namespace PicturedFactCollision

/-- Two disjoint sets of legitimate reference images may contain different
paraphrases. The paired test requires one output accepted on each page. -/
structure Pair where
  sameInput : Bool
  bothCorrect : Bool
  disjoint : Bool
  sound : sameInput = true → disjoint = true → bothCorrect = false

def collision (p : Pair) := p.sameInput && p.disjoint

theorem deterministic_obstruction {I O : Type} (write : I → O)
    (a b : I) (A B : O → Prop) (h : a = b)
    (sep : ∀ y, A y → ¬ B y) : ¬ (A (write a) ∧ B (write b)) := by
  subst b
  intro pair
  exact sep _ pair.1 pair.2

def count (p : Pair → Bool) : List Pair → Nat
  | [] => 0
  | x :: xs => (if p x then 1 else 0) + count p xs

theorem paired_capacity (xs : List Pair) :
    count collision xs + count Pair.bothCorrect xs ≤ xs.length := by
  induction xs with
  | nil => simp [count]
  | cons x xs ih =>
    have sound := x.sound
    cases hc : x.sameInput <;> cases hd : x.disjoint <;> cases hb : x.bothCorrect <;>
      simp_all [count, collision] <;> omega
end PicturedFactCollision
