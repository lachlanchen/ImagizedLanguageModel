import Std

namespace PairedAnswerMass
variable {G : Type} [DecidableEq G]
abbrev Paths (G : Type) := List (List G × Nat)

def total : Paths G → Nat
  | [] => 0
  | (_,w)::xs => w + total xs

def vote (a : Option G) : Paths G → Nat
  | [] => 0
  | (s,w)::xs => (if s.head? = a then w else 0) + vote a xs

def consume (g : G) : Paths G → Paths G
  | [] => []
  | ([],_)::xs => consume g xs
  | (h::s,w)::xs => if h=g then (s,w)::consume g xs else consume g xs

theorem vote_le (a : Option G) (xs : Paths G) : vote a xs ≤ total xs := by
  induction xs with
  | nil => simp [vote,total]
  | cons x xs ih =>
    obtain ⟨s,w⟩ := x
    by_cases h : s.head?=a <;> simp [vote,total,h] <;> omega

theorem consume_le (a : G) (xs : Paths G) : total (consume a xs) ≤ total xs := by
  induction xs with
  | nil => simp [consume,total]
  | cons x xs ih =>
    obtain ⟨s,w⟩ := x
    cases s with
    | nil => simp [consume,total]; omega
    | cons g s =>
      by_cases h : g=a <;> simp [consume,total,h] <;> omega

def strict (s : List G) (xs : Paths G) :=
  ∀ a : Option G, a ≠ s.head? → vote a xs < vote s.head? xs

def writes : List G → Paths G → Prop
  | [], xs => strict [] xs
  | a::s, xs => strict (a::s) xs ∧ writes s (consume a xs)

/-- One complete target atom outweighing all competing paths suffices for
every greedy decision, including stopping, with arbitrary shared prefixes. -/
theorem majority_writes (s : List G) (w : Nat) (xs : Paths G)
    (h : total xs < w) : writes s ((s,w)::xs) := by
  have wins (v : List G) (others : Paths G) (hw : total others < w) :
      strict v ((v,w)::others) := by
    intro a diff
    have hb := vote_le a others
    simp [vote, Ne.symm diff]
    omega
  induction s generalizing xs with
  | nil => exact wins [] xs h
  | cons a s ih =>
    constructor
    · exact wins (a::s) xs h
    · simp [consume]
      exact ih _ (Nat.lt_of_le_of_lt (consume_le a xs) h)

#print axioms majority_writes
end PairedAnswerMass
