import Std

namespace SemanticWritingRisk

/-- Count viable prefixes by generation depth in a finite batch of actual
rollouts. A prefix is viable iff some completion within the length limit can
express the requested facts and stop. Mark an early stop as an absorbing
completion. The step's loss counts newly irreversible failures. -/
def lost (viable : Nat → Nat) : Nat → Nat
  | 0 => 0
  | n+1 => lost viable n + (viable n - viable (n+1))

theorem countFirstFailures (viable : Nat → Nat)
    (nested : ∀ i, viable (i+1) ≤ viable i) (limit : Nat) :
    viable limit + lost viable limit = viable 0 := by
  induction limit with
  | zero => simp [lost]
  | succ n ih =>
    have h := nested n
    simp only [lost]
    omega

/-- With two contradictory records, suppose sentence totals are measured in
equal mass units N. Under the second record the two sets of faithful sentences
are disjoint. If switching inputs can move at most d units into the first
record's faithful set, combined successes cannot exceed N+d. This uses no
assumption that a single reference is the only acceptable wording. -/
theorem counterfactualBound (N d successA successB massAunderB : Nat)
    (disjoint : massAunderB + successB ≤ N)
    (shift : successA ≤ massAunderB + d) :
    successA + successB ≤ N + d := by omega

/-- If a faithful completed candidate has an ideal score advantage above twice
the uniform error of the learned scorer, its empirical maximizer is faithful.
The finite beam may fail to contain such a witness. Scores are scaled integers. -/
theorem faithfulRerank {Y : Type} (faithful eligible : Y → Prop)
    (ideal learned : Y → Int) (err : Int) (w chosen : Y)
    (witness : eligible w) (inside : eligible chosen)
    (separation : ∀ y, eligible y → ¬ faithful y → ideal y + 2 * err < ideal w)
    (lower : ∀ y, eligible y → ideal y - err ≤ learned y)
    (upper : ∀ y, eligible y → learned y ≤ ideal y + err)
    (maximal : ∀ y, eligible y → learned y ≤ learned chosen) : faithful chosen := by
  apply Classical.byContradiction
  intro bad
  have h₁ := separation chosen inside bad
  have h₂ := lower w witness
  have h₃ := upper chosen inside
  have h₄ := maximal w witness
  omega

end SemanticWritingRisk
