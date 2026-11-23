import Std
namespace VisualPrefixMass

abbrev Corpus (α : Type) := List (List α × Nat)
def total {α : Type} : Corpus α → Nat
  | [] => 0
  | (_,w) :: xs => w + total xs

def terminal {α : Type} : Corpus α → Nat
  | [] => 0
  | ([],w) :: xs => w + terminal xs
  | (_ :: _,_) :: xs => terminal xs

/-- Conditioning consumes one physical glyph and retains every compatible
occurrence, including multiple source positions with identical bitmaps. -/
def step {α : Type} [DecidableEq α] (a : α) : Corpus α → Corpus α
  | [] => []
  | ([],_) :: xs => step a xs
  | (b :: bs,w) :: xs => if b = a then (bs,w) :: step a xs else step a xs

def afterPrefix {α : Type} [DecidableEq α] (xs : Corpus α) : List α → Corpus α
  | [] => xs
  | a :: as => afterPrefix (step a xs) as

theorem append_prefix {α : Type} [DecidableEq α] (xs : Corpus α) (p q : List α) :
    afterPrefix xs (p ++ q) = afterPrefix (afterPrefix xs p) q := by
  induction p generalizing xs with
  | nil => rfl
  | cons a p ih => exact ih (step a xs)

/-- Terminal mass after a prefix equals the mass of identical complete
outputs in the original corpus. -/
def exactMass {α : Type} [DecidableEq α] (p : List α) : Corpus α → Nat
  | [] => 0
  | (s,w)::xs => (if s = p then w else 0) + exactMass p xs

theorem consume {α : Type} [DecidableEq α] (a : α) (p : List α) (xs : Corpus α) :
    exactMass p (step a xs) = exactMass (a::p) xs := by
  induction xs with
  | nil => rfl
  | cons x xs ih =>
    obtain ⟨s,w⟩ := x
    cases s with
    | nil => simp [step,exactMass,ih]
    | cons b bs =>
      by_cases h : b = a
      · subst b
        simp [step,exactMass,ih]
      · simp [step,exactMass,h,ih]

theorem empty_exact {α : Type} [DecidableEq α] (xs : Corpus α) :
    terminal xs = exactMass [] xs := by
  induction xs with
  | nil => rfl
  | cons x xs ih =>
    obtain ⟨s,w⟩ := x
    cases s <;> simp [terminal,exactMass,ih]

theorem terminal_mass {α : Type} [DecidableEq α] (p : List α) (xs : Corpus α) :
    terminal (afterPrefix xs p) = exactMass p xs := by
  induction p generalizing xs with
  | nil => exact empty_exact xs
  | cons a p ih => exact (ih (step a xs)).trans (consume a p xs)

#print axioms append_prefix
#print axioms terminal_mass
end VisualPrefixMass
