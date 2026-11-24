import Std

namespace ClozeStateRisk

/-- Deterministic continuation after the source has been compressed into a state. -/
def rollout (step : S → Y × S) : Nat → S → List Y
  | 0, _ => []
  | n+1, s => (step s).1 :: rollout step n (step s).2

theorem sameFuture (step : S → Y × S) (n : Nat) (h : a = b) :
    rollout step n a = rollout step n b := by rw [h]

/-- Sum of correct suffixes when cases are grouped by the state after two
emissions. Each empirical histogram may contain any number of distinct suffixes. -/
def hits (hist : S → Y → Nat) (choose : S → Y) : List S → Nat
  | [] => 0
  | s :: ss => hist s (choose s) + hits hist choose ss

/-- Choosing the most frequent desired suffix separately for each state gives
a tight upper bound on every deterministic decoder using only that state. -/
theorem optimal (hist : S → Y → Nat) (mode choose : S → Y) (states : List S)
    (best : ∀ s ∈ states, ∀ y, hist s y ≤ hist s (mode s)) :
    hits hist choose states ≤ hits hist mode states := by
  induction states with
  | nil => simp [hits]
  | cons s ss ih =>
    have first := best s (by simp) (choose s)
    have rest := ih (fun t h => best t (List.mem_cons_of_mem _ h))
    exact Nat.add_le_add first rest

/-- Equality requires a modal choice in every observed state. The condition
is exact even when several suffixes tie for the most frequent outcome. -/
theorem equality (hist : S → Y → Nat) (mode choose : S → Y) (states : List S)
    (best : ∀ s ∈ states, ∀ y, hist s y ≤ hist s (mode s)) :
    hits hist choose states = hits hist mode states ↔
      ∀ s ∈ states, hist s (choose s) = hist s (mode s) := by
  induction states with
  | nil => simp [hits]
  | cons s ss ih =>
    have bound := best s (by simp) (choose s)
    have tailbest := fun t (h : t ∈ ss) => best t (List.mem_cons_of_mem _ h)
    have tailbound := optimal hist mode choose ss tailbest
    have tailiff := ih tailbest
    simp only [hits, List.forall_mem_cons]
    constructor
    · intro eq
      exact ⟨by omega, tailiff.mp (by omega)⟩
    · rintro ⟨head, tail⟩
      rw [head, tailiff.mpr tail]

end ClozeStateRisk
