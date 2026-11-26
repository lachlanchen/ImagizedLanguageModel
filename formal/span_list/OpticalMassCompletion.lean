import Std
namespace OpticalMassCompletion

/-- Each entry is a known teacher lower mass and a student's proposed mass,
expressed in units of the same finite denominator. -/
abbrev Row := Nat × Nat

def floorSum : List Row → Nat
  | [] => 0
  | (l,_)::xs => l + floorSum xs

def studentSum : List Row → Nat
  | [] => 0
  | (_,q)::xs => q + studentSum xs

def capacity : List Row → Nat
  | [] => 0
  | (l,q)::xs => (q-l) + capacity xs

def deficit : List Row → Nat
  | [] => 0
  | (l,q)::xs => (l-q) + deficit xs

/-- Feasible completed masses and their total excess over the student's mass.
For equal total mass, the excess is total variation in integer units. -/
inductive Completion : List Row → Nat → Nat → Prop
  | nil : Completion [] 0 0
  | cons {xs total excess l q p} : l ≤ p → Completion xs total excess →
      Completion ((l,q)::xs) (p+total) ((p-q)+excess)

theorem lower {xs total excess} (h : Completion xs total excess) :
    deficit xs ≤ excess := by
  induction h with
  | nil => simp [deficit]
  | cons h _ ih => simp only [deficit]; omega

theorem mass_balance (xs : List Row) :
    studentSum xs + deficit xs = floorSum xs + capacity xs := by
  induction xs with
  | nil => rfl
  | cons a xs ih =>
    rcases a with ⟨l,q⟩
    simp only [studentSum, deficit, floorSum, capacity]
    omega

/-- Distribute unassigned mass using spare student capacity. No coordinate
crosses its proposed mass unless its observed floor already exceeded it. -/
theorem fill (xs : List Row) (r : Nat) (h : r ≤ capacity xs) :
    Completion xs (floorSum xs + r) (deficit xs) := by
  induction xs generalizing r with
  | nil =>
    simp [capacity] at h
    subst r
    exact Completion.nil
  | cons a xs ih =>
    rcases a with ⟨l,q⟩
    let take := min r (q-l)
    have ht : take ≤ r := Nat.min_le_left _ _
    have hc : take ≤ q-l := Nat.min_le_right _ _
    have rest : r-take ≤ capacity xs := by
      simp only [capacity] at h
      dsimp [take]; omega
    have sub := ih (r-take) rest
    have made := Completion.cons (q:=q) (l:=l) (p:=l+take) (by omega) sub
    have eq₁ : (l+take)+(floorSum xs+(r-take)) = floorSum ((l,q)::xs)+r := by
      simp only [floorSum]; omega
    have eq₂ : (l+take-q)+deficit xs = deficit ((l,q)::xs) := by
      simp only [deficit]; omega
    rwa [eq₁,eq₂] at made

/-- The lower bound is attained by a teacher completion having the same
total mass as the deployed student's distribution. -/
theorem least_unavoidable (xs : List Row) (h : floorSum xs ≤ studentSum xs) :
    Completion xs (studentSum xs) (deficit xs) ∧
      (∀ e, Completion xs (studentSum xs) e → deficit xs ≤ e) := by
  have balance := mass_balance xs
  have made := fill xs (studentSum xs - floorSum xs) (by omega)
  have e : floorSum xs + (studentSum xs-floorSum xs) = studentSum xs := by omega
  exact ⟨e ▸ made, fun _ h' => lower h'⟩

#print axioms least_unavoidable
end OpticalMassCompletion
