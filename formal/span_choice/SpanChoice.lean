import Std

namespace SpanChoice
variable {Glyphs : Type} [DecidableEq Glyphs]

/-- Votes for identical physical strips, including identical strips at different locations. -/
def mass (y : Glyphs) : List (Glyphs × Nat) → Nat
  | [] => 0
  | (v, w) :: tail => (if v = y then w else 0) + mass y tail

def total : List (Glyphs × Nat) → Nat
  | [] => 0
  | (_, w) :: tail => w + total tail

theorem disjoint_mass (a b : Glyphs) (distinct : a ≠ b) (xs : List (Glyphs × Nat)) :
    mass a xs + mass b xs ≤ total xs := by
  induction xs with
  | nil => simp [mass, total]
  | cons x xs ih =>
    obtain ⟨v,w⟩ := x
    by_cases ha : v = a
    · have hb : v ≠ b := fun h => distinct (ha.symm.trans h)
      simp [mass,total,ha,hb,distinct] <;> omega
    · by_cases hb : v = b <;> simp [mass,total,ha,hb,Ne.symm distinct] <;> omega

/-- A pooled wrong winner charges at least half of total mass to other strips. -/
theorem error_bound (truth chosen : Glyphs) (xs : List (Glyphs × Nat))
    (winner : mass truth xs ≤ mass chosen xs) (wrong : chosen ≠ truth) :
    total xs ≤ 2 * (total xs - mass truth xs) := by
  have h := disjoint_mass chosen truth wrong xs
  omega

theorem majority_recovers (truth chosen : Glyphs) (xs : List (Glyphs × Nat))
    (winner : mass truth xs ≤ mass chosen xs) (strong : total xs < 2 * mass truth xs) :
    chosen = truth := by
  by_cases eq : chosen = truth
  · exact eq
  · have h := disjoint_mass chosen truth eq xs
    omega

omit [DecidableEq Glyphs] in
theorem uncovered (truth chosen : Glyphs) (options : List Glyphs)
    (produced : chosen ∈ options) (absent : truth ∉ options) : chosen ≠ truth :=
  fun h => absent (h ▸ produced)

example : mass "shared" [("other",4),("shared",3),("shared",3)] = 6 := by decide

#print axioms disjoint_mass
#print axioms error_bound
#print axioms majority_recovers
#print axioms uncovered
end SpanChoice

