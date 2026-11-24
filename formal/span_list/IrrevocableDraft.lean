import Std

namespace IrrevocableDraft

/-- A slot carries its reference image, its committed image (if any), and the
images individually writable there. The latter may overestimate neural reachability. -/
structure Slot (G : Type) where
  reference : G
  committed : Option G
  writable : G → Prop

def permitted (s : Slot G) (x : G) : Prop :=
  match s.committed with
  | some p => x = p
  | none => s.writable x

inductive Completes : List (Slot G) → List G → Prop
  | nil : Completes [] []
  | cons (p : permitted s x) (rest : Completes ss xs) :
      Completes (s :: ss) (x :: xs)

/-- Necessary and sufficient for exact reconstruction in the slotwise relaxation.
A negative result is a certificate against every decoder that keeps these commitments. -/
theorem exact_iff (slots : List (Slot G)) :
    Completes slots (slots.map Slot.reference) ↔
    ∀ s ∈ slots, permitted s s.reference := by
  induction slots with
  | nil => simp [Completes.nil]
  | cons s ss ih =>
    constructor
    · intro h
      cases h with
      | cons p tail =>
        intro t mem
        cases List.mem_cons.mp mem with
        | inl eq => simpa [eq] using p
        | inr inside => exact ih.mp tail t inside
    · intro h
      exact .cons (h s (by simp)) (ih.mpr (fun t mem =>
        h t (List.mem_cons_of_mem _ mem)))

theorem wrong_commit (slots : List (Slot G)) (s : Slot G) (p : G)
    (inside : s ∈ slots) (fixed : s.committed = some p) (wrong : p ≠ s.reference) :
    ¬ Completes slots (slots.map Slot.reference) := by
  intro exactDraft
  have eq := (exact_iff slots).mp exactDraft s inside
  simp [permitted, fixed] at eq
  exact wrong eq.symm

end IrrevocableDraft
