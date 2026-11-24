import Std

namespace AnonymousCast

/-- The marked reading query stays in the written context. Names have an
    external identity that only the copying and pooling operations can see. -/
inductive Cell (G E : Type) where
  | printed (glyph : G)
  | name (entity : E)

def erase {G E} : Cell G E → Option G
  | .printed g => some g
  | .name _ => none

def rename {G E F} (f : E → F) : Cell G E → Cell G F
  | .printed g => .printed g
  | .name e => .name (f e)

theorem identicalContext {G E F} (f : E → F) (page : List (Cell G E)) :
    (page.map (rename f)).map erase = page.map erase := by
  induction page with
  | nil => rfl
  | cons h t ih => cases h <;> simp [rename, erase, ih]

/-- A scorer may depend arbitrarily on the full anonymized context and the
    marked query location. It therefore assigns identical occurrence weights. -/
theorem identicalScores {G E F W} (f : E → F) (page : List (Cell G E))
    (query : Nat × Nat) (score : List (Option G) → Nat × Nat → W) :
    score ((page.map (rename f)).map erase) query = score (page.map erase) query := by
  rw [identicalContext]

def mass {E W : Type} [DecidableEq E] [OfNat W 0] [Add W]
    (key : E) : List (E × W) → W
  | [] => 0
  | (e,w)::xs => (if e = key then w else 0) + mass key xs

/-- Summing over visual occurrences commutes with any injective name change.
    This argument also applies to nonnegative real weights, without requiring
    approximate equality of two floating-point neural networks. -/
theorem renamedMass {E F W : Type} [DecidableEq E] [DecidableEq F]
    [OfNat W 0] [Add W] (f : E → F) (inj : ∀ a b, f a = f b → a = b)
    (key : E) (xs : List (E × W)) :
    mass (f key) (xs.map (fun p => (f p.1,p.2))) = mass key xs := by
  induction xs with
  | nil => rfl
  | cons h tail ih =>
    obtain ⟨e,w⟩ := h
    by_cases h : e = key
    · subst e; simp [mass, ih]
    · have hn : f e ≠ f key := fun eq => h (inj e key eq)
      simp [mass,h,hn,ih]

#print axioms identicalScores
#print axioms renamedMass
end AnonymousCast

namespace LocalReading

/-- A pair asks for opposite referents. If the visible statistics of the page
    coincide, no deterministic readout of those statistics can pass the pair. -/
theorem oppositePairRequiresEvidence {Page View Answer : Type}
    (visible : Page → View) (read : View → Answer)
    (x y : Page) (a b : Answer) (different : a ≠ b)
    (both : read (visible x) = a ∧ read (visible y) = b) :
    visible x ≠ visible y := by
  intro collision
  exact different (both.1.symm.trans ((congrArg read collision).trans both.2))

#print axioms oppositePairRequiresEvidence
end LocalReading
