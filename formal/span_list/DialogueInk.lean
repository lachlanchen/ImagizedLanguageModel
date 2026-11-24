import Std

namespace DialogueInk

/-- A physical glyph can be a bitmap. The argument needs only equality. -/
inductive Emit (Glyph : Type) where
  | point (surface : List Glyph)
  | spell (surface : List Glyph)

def admissible {G : Type} (visible : List (List G)) (palette : List G) : Emit G → Prop
  | .point w => w ∈ visible ∧ w ≠ []
  | .spell w => (∀ g ∈ w, g ∈ palette) ∧ w ≠ [] ∧ w.length ≤ 24

def surface {G : Type} : Emit G → List G
  | .point w | .spell w => w

/-- Any reply of at most 16 nonempty words can be emitted in its requested
order when each is either a visible candidate or spellable in the print palette.
This is an interface support result before applying learned greedy decisions. -/
theorem reachable {G : Type} (visible : List (List G)) (palette : List G)
    (words : List (List G)) (size : words.length ≤ 16)
    (available : ∀ w ∈ words, admissible visible palette (.point w) ∨
                              admissible visible palette (.spell w)) :
    ∃ actions : List (Emit G), actions.map surface = words ∧ actions.length ≤ 16 ∧
      ∀ a ∈ actions, admissible visible palette a := by
  let _ : DecidablePred (fun w => admissible visible palette (.point w)) := fun _ => Classical.propDecidable _
  let choose : List G → Emit G := fun w =>
    if admissible visible palette (.point w) then .point w else .spell w
  have emits (w : List G) : surface (choose w) = w := by
    by_cases h : admissible visible palette (.point w) <;> simp [choose,h,surface]
  refine ⟨words.map choose, ?_, by simpa using size, ?_⟩
  · simp [List.map_map, Function.comp_def, emits]
  · intro a ha
    obtain ⟨w,hw,rfl⟩ := List.mem_map.mp ha
    by_cases h : admissible visible palette (.point w)
    · simpa [choose,h] using h
    · simpa [choose,h] using (available w hw).resolve_left h

/-- A marginal pointer loss sums the likelihood of every source occurrence
of the same requested physical word. Appending disjoint candidate lists adds
their mass, including repeated occurrences on opposite sides. -/
def mass {G : Type} [DecidableEq G] (target : G) : List (G × Nat) → Nat
  | [] => 0
  | (g,n)::rest => (if g = target then n else 0) + mass target rest

theorem pointer_mass_append {G : Type} [DecidableEq G] (target : G) (x y : List (G × Nat)) :
    mass target (x ++ y) = mass target x + mass target y := by
  induction x with
  | nil => simp [mass]
  | cons h t ih =>
    obtain ⟨g,n⟩ := h
    simp [mass, ih, Nat.add_assoc]

#print axioms reachable
#print axioms pointer_mass_append
end DialogueInk
