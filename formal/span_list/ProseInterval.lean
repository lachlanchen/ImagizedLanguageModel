import Std

namespace ProseInterval

variable {W : Type} [OfNat W 0] [Add W] [Mul W]

structure Cell (W : Type) where
  opening : W
  interior : W
  closing : W

/-- Total weight of emitting an end in the remaining positions for a span already open. -/
def endings : List (Cell W) → W
  | [] => 0
  | x :: xs => x.interior * (x.closing + endings xs)

/-- Every possible starting position, with all reachable ends. -/
def intervals : List (Cell W) → W
  | [] => 0
  | x :: xs => x.opening * endings (x :: xs) + intervals xs

/-- Carry only the combined weight of intervals open at the previous cell. -/
def scan (openWeight : W) : List (Cell W) → W
  | [] => 0
  | x :: xs =>
    let active := (openWeight + x.opening) * x.interior
    active * x.closing + scan active xs

theorem allEndpoints
    (plusAssoc : ∀ a b c : W, (a + b) + c = a + (b + c))
    (timesAssoc : ∀ a b c : W, (a * b) * c = a * (b * c))
    (zeroR : ∀ a : W, a * 0 = 0)
    (zeroPlus : ∀ a : W, 0 + a = a)
    (leftDist : ∀ a b c : W, a * (b + c) = a * b + a * c)
    (rightDist : ∀ a b c : W, (a + b) * c = a * c + b * c)
    (xs : List (Cell W)) (p : W) :
    scan p xs = p * endings xs + intervals xs := by
  induction xs generalizing p with
  | nil => simp [scan, endings, intervals, zeroR, zeroPlus]
  | cons x xs ih =>
    simp only [scan, ih, intervals, endings]
    rw [← plusAssoc, ← leftDist, timesAssoc, rightDist, plusAssoc]

#print axioms allEndpoints
end ProseInterval

namespace ProseInterval

/-- A locator with no referent argument returns one interval on a fixed page and
    property. It cannot answer both referents if their credited intervals differ. -/
theorem referentRequired {Page Role Span : Type}
    (locate : Page → Role → Span) (page : Page) (field : Role) (a b : Span) (h : a ≠ b) :
    ¬ (locate page field = a ∧ locate page field = b) := by
  intro result
  exact h (result.1.symm.trans result.2)

end ProseInterval

namespace ProseInterval

/-- Counts of two disjoint correct intervals among cases with the same visible input.
    An unrelated interval or abstention is wrong on both groups. -/
def mistaken (counts : Nat × Nat) : Option Bool → Nat
  | some false => counts.2
  | some true => counts.1
  | none => counts.1 + counts.2

def majority (counts : Nat × Nat) : Option Bool :=
  if counts.2 ≤ counts.1 then some false else some true

theorem majorityLoss (c : Nat × Nat) :
    mistaken c (majority c) = min c.1 c.2 := by
  unfold majority
  split <;> simp only [mistaken] <;> omega

def observedErrors {X : Type} (counts : X → Nat × Nat) (choose : X → Option Bool) :
    List X → Nat
  | [] => 0
  | x :: xs => mistaken (counts x) (choose x) + observedErrors counts choose xs

/-- Exact empirical zero-one risk floor. The count function pools cases by
    their observed input; the chosen function may otherwise be arbitrary. -/
theorem observedRiskFloor {X : Type} (counts : X → Nat × Nat) (choose : X → Option Bool)
    (inputs : List X) :
    observedErrors counts (fun x => majority (counts x)) inputs ≤
    observedErrors counts choose inputs := by
  induction inputs with
  | nil => exact Nat.le_refl 0
  | cons x xs ih =>
    simp only [observedErrors]
    apply Nat.add_le_add _ ih
    rw [majorityLoss]
    cases h : choose x with
    | none => simp only [mistaken]; omega
    | some decision => cases decision <;> simp only [mistaken] <;> omega

#print axioms observedRiskFloor

end ProseInterval
