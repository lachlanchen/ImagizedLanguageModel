import Std

namespace FirstActionRisk

def mass : List Nat → Nat
  | [] => 0
  | a :: xs => a + mass xs

/-- Sum over a list of nonnegative action masses. Repeated positions bearing
the same physical glyph must be aggregated before applying this result. -/
theorem rival_le_sum (rival : Nat) (rest : List Nat) (h : rival ∈ rest) :
    rival ≤ mass rest := by
  induction rest with
  | nil => simp at h
  | cons a xs ih =>
    simp only [List.mem_cons] at h
    rcases h with rfl | h
    · simp [mass]
    · have := ih h
      simp only [mass]
      omega

/-- A wrong greedy first choice entails at least half of the total mass
outside the intended action. Equal masses allow an incorrect tie-break. -/
theorem wrong_cost (gold rival : Nat) (others : List Nat)
    (present : rival ∈ others) (wins : gold ≤ rival) :
    gold + mass others ≤ 2 * mass others := by
  have := rival_le_sum rival others present
  omega

/-- Partition stopped-answer outcomes using two Boolean tests. -/
def exact (cases : List (Bool × Bool)) : Nat :=
  (cases.filter (fun x => x.1 && x.2)).length

def firstRight (cases : List (Bool × Bool)) : Nat :=
  (cases.filter (fun x => x.1)).length

def suffixMiss (cases : List (Bool × Bool)) : Nat :=
  (cases.filter (fun x => x.1 && !x.2)).length

theorem partition (cases : List (Bool × Bool)) :
    exact cases + suffixMiss cases = firstRight cases := by
  induction cases with
  | nil => rfl
  | cons x xs ih =>
    obtain ⟨a,b⟩ := x
    cases a <;> cases b <;>
      simp_all [exact, suffixMiss, firstRight, List.filter_cons] <;> omega

theorem gain_iff (a b : List (Bool × Bool)) :
    exact b > exact a ↔ firstRight b + suffixMiss a > firstRight a + suffixMiss b := by
  have := partition a
  have := partition b
  omega

#print axioms wrong_cost
#print axioms gain_iff
end FirstActionRisk
