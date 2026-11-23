import Std

namespace StoppedPreference

/-- Each pair contains the old and new log score for one action along a
specified prefix. Integers are log scores in an arbitrary common unit. -/
def drift : List (Int × Int) → Int
  | [] => 0
  | (old, updated) :: rest => updated - old + drift rest

def bounded (ε : Int) (actions : List (Int × Int)) : Prop :=
  ∀ p ∈ actions, -ε ≤ p.2 - p.1 ∧ p.2 - p.1 ≤ ε

theorem drift_bounds (ε : Int) (xs : List (Int × Int)) (h : bounded ε xs) :
    -(Int.ofNat xs.length * ε) ≤ drift xs ∧ drift xs ≤ Int.ofNat xs.length * ε := by
  induction xs with
  | nil => simp [drift]
  | cons p rest ih =>
    have step := h p (by simp)
    have tail := ih (fun x hx => h x (by simp [hx]))
    obtain ⟨a,b⟩ := p
    have hs : Int.ofNat (rest.length + 1) = Int.ofNat rest.length + 1 := rfl
    simp only [List.length_cons, hs, Int.add_mul, Int.one_mul, drift]
    simp only at step
    omega

/-- Changes on the common prefix of the two stopped replies cancel. The
two lists below start at the first divergent action and include the stop. -/
theorem necessary_reversal (oldRef oldRival common ε : Int)
    (ref rival : List (Int × Int)) (hr : bounded ε ref) (hw : bounded ε rival)
    (newWins : oldRival + common + drift rival ≤ oldRef + common + drift ref) :
    oldRival - oldRef ≤ (Int.ofNat ref.length + Int.ofNat rival.length) * ε := by
  have r := drift_bounds ε ref hr
  have w := drift_bounds ε rival hw
  rw [Int.add_mul]
  omega

#print axioms necessary_reversal
end StoppedPreference
