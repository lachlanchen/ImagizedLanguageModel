import Std

namespace InkFeedbackAgreement

/-- The command includes stop. Both generators begin at the same state and
execute at most fuel choices. State may include decoder caches and position. -/
def run {S C I : Type} (choose : S → Option C) (paint : C → I)
    (feedback : S → I → S) (s : S) : Nat → List I × Bool
  | 0 => ([], false)
  | n + 1 => match choose s with
    | none => ([], true)
    | some c =>
      let i := paint c
      let (tail, stopped) := run choose paint feedback (feedback s i) n
      (i :: tail, stopped)

/-- A uniform bound on logits is unnecessary. Exactness only on choices
reachable by the reference trajectory suffices for identical images and stop. -/
def agrees {S C I : Type} (choose : S → Option C) (p q : C → I)
    (feedback : S → I → S) (s : S) : Nat → Prop
  | 0 => True
  | n + 1 => match choose s with
    | none => True
    | some c => p c = q c ∧ agrees choose p q feedback (feedback s (p c)) n

theorem trajectory {S C I : Type} (choose : S → Option C) (p q : C → I)
    (feedback : S → I → S) (s : S) (n : Nat)
    (h : agrees choose p q feedback s n) :
    run choose p feedback s n = run choose q feedback s n := by
  induction n generalizing s with
  | zero => rfl
  | succ n ih =>
    cases hc : choose s with
    | none => simp [run, hc]
    | some c =>
      simp [agrees, hc] at h
      obtain ⟨same, rest⟩ := h
      have tail := ih (feedback s (p c)) rest
      simp only [run, hc]
      rw [← same, tail]

#print axioms trajectory
end InkFeedbackAgreement
