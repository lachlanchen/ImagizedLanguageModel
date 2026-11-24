import Std

namespace VisualTokenRisk

/-- Total loss over a finite sample, with no distributional assumption. -/
def risk {X Y : Type} (loss : Y → Y → Nat) (p : X → Y) (d : List (X × Y)) : Nat :=
  (d.map (fun s => loss (p s.1) s.2)).foldl Nat.add 0

theorem renderRisk {X I Y : Type} (render : X → I) (read : I → X)
    (recover : ∀ x, read (render x) = x) (loss : Y → Y → Nat) (p : X → Y)
    (d : List (X × Y)) :
    risk loss (fun i => p (read i)) (d.map (fun s => (render s.1, s.2))) =
      risk loss p d := by
  simp [risk, List.map_map, Function.comp_def, recover]

theorem pullbackRisk {X I Y : Type} (render : X → I) (loss : Y → Y → Nat)
    (q : I → Y) (d : List (X × Y)) :
    risk loss q (d.map (fun s => (render s.1, s.2))) =
      risk loss (fun x => q (render x)) d := by
  simp [risk, List.map_map, Function.comp_def]

/-- Two spellings collapsed to identical ink necessarily receive the same
prediction. If their requested answers differ, at least one answer is wrong. -/
theorem collision {X I Y : Type} (render : X → I) (q : I → Y) (x z : X)
    (a b : Y) (same : render x = render z) (different : a ≠ b) :
    q (render x) ≠ a ∨ q (render z) ≠ b := by
  by_cases h : q (render x) = a
  · right
    intro k
    exact different (h.symm.trans ((congrArg q same).trans k))
  · exact Or.inl h

/-- At matched performance, inherited training and new training both enter a
cost claim. The factor and energy unit must be fixed across treatments. -/
theorem inheritedBudget (earlier added baseline factor : Nat)
    (paid : (earlier + added) * factor ≤ baseline) :
    added * factor ≤ baseline :=
  Nat.le_trans (Nat.mul_le_mul_right factor (Nat.le_add_left added earlier)) paid

#print axioms renderRisk
#print axioms pullbackRisk
#print axioms collision
#print axioms inheritedBudget

end VisualTokenRisk

namespace WrittenReply

/-- A step reads the preceding *emitted words*. Its output is the finished word
image or the decision to stop. An erroneous unterminated word can be represented
by a fresh failure symbol outside the reference alphabet. -/
def run {W : Type} (step : List W → Option W) (history : List W) :
    Nat → Option (List W)
  | 0 => none
  | n+1 => match step history with
    | none => some []
    | some w => (run step (history ++ [w]) n).map (fun tail => w :: tail)

def correctAlong {W : Type} (step : List W → Option W) (p : List W) : List W → Prop
  | [] => step p = none
  | w :: ws => step p = some w ∧ correctAlong step (p ++ [w]) ws

private theorem consMap {W : Type} (x y : W) (r : Option (List W)) (ys : List W) :
    r.map (fun tail => x :: tail) = some (y :: ys) ↔ x = y ∧ r = some ys := by
  cases r <;> simp

/-- Exact certificate for a complete free-running answer under deterministic
feedback. Checking the stop on the gold history matters as much as checking a word. -/
theorem exactIff {W : Type} (step : List W → Option W) (target history : List W) :
    run step history (target.length + 1) = some target ↔
    correctAlong step history target := by
  induction target generalizing history with
  | nil =>
    cases h : step history <;> simp [run, correctAlong, h]
  | cons w ws ih =>
    cases h : step history with
    | none => simp [run, correctAlong, h]
    | some v =>
      simp only [List.length_cons, run, h, correctAlong, Option.some.injEq]
      change (run step (history ++ [v]) (ws.length + 1)).map (fun tail => v :: tail) = some (w :: ws) ↔ v = w ∧ correctAlong step (history ++ [w]) ws
      rw [consMap, ih]
      constructor <;> intro ⟨eq,good⟩ <;> subst v <;> exact ⟨rfl,good⟩

/-- For scores in common fixed-point units, a strict margin exceeding twice
the per-score perturbation preserves the winning action through a style change. -/
theorem preserveMargin {A : Type} (clean changed : A → Int) (a b : A) (e : Int)
    (margin : clean b + 2*e < clean a)
    (aLower : clean a - e ≤ changed a) (bUpper : changed b ≤ clean b + e) :
    changed b < changed a := by omega

#print axioms exactIff
#print axioms preserveMargin
end WrittenReply
