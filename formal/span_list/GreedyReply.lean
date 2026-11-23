import Std

namespace GreedyReply

/-- A reply is a finite sequence of ink cells followed by a stop.
History is stored most-recent-cell first, so extending it costs one cons. -/
def decode {Ink : Type} (step : List Ink → Option Ink) :
    Nat → List Ink → Option (List Ink)
  | 0, _ => none
  | fuel + 1, history =>
    match step history with
    | none => some []
    | some x => (decode step fuel (x :: history)).map (x :: ·)

def follows {Ink : Type} (step : List Ink → Option Ink) :
    List Ink → List Ink → Prop
  | h, [] => step h = none
  | h, x :: xs => step h = some x ∧ follows step (x :: h) xs

theorem exact_stopped_iff {Ink : Type} (step : List Ink → Option Ink)
    (h y : List Ink) :
    decode step (y.length + 1) h = some y ↔ follows step h y := by
  induction y generalizing h with
  | nil =>
    cases e : step h <;> simp [decode, follows, e]
  | cons a tail ih =>
    change (match step h with
      | none => some []
      | some b => (decode step (tail.length + 1) (b :: h)).map (b :: ·)) =
        some (a :: tail) ↔ step h = some a ∧ follows step (a :: h) tail
    cases e : step h with
    | none => simp
    | some b =>
      by_cases same : b = a
      · subst b
        specialize ih (a :: h)
        cases r : decode step (tail.length + 1) (a :: h) <;> simp_all
      · cases decode step (tail.length + 1) (b :: h) <;> simp [same]

/-- An argmax action must be the intended one if the reference margin
exceeds twice a bound on each action's score perturbation. Integer scores
may denote fixed-point logits in a common unit. -/
theorem margin_forces_choice {A : Type} (reference actual : A → Int)
    (target chosen : A) (ε : Int)
    (lower : reference target - ε ≤ actual target)
    (upper : actual chosen ≤ reference chosen + ε)
    (separated : ∀ a, a ≠ target → reference a + 2 * ε < reference target)
    (winner : actual target ≤ actual chosen) : chosen = target := by
  by_cases h : chosen = target
  · exact h
  · have gap := separated chosen h
    omega

#print axioms exact_stopped_iff
#print axioms margin_forces_choice
end GreedyReply
