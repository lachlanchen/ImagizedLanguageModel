import Std

namespace MatchedReadingCost

structure Checkpoint where
  heldCorrect : Nat
  devCorrect : Nat
  energyMJ : Nat -- millijoules, including measured preparation

def qualifies (held dev percent : Nat) (r : Checkpoint) : Bool :=
  decide (held > 0 ∧ dev > 0 ∧ percent * held ≤ 100 * r.heldCorrect ∧
    percent * dev ≤ 100 * r.devCorrect)

def minimum (ok : Checkpoint → Bool) : List Checkpoint → Option Nat
  | [] => none
  | r :: rs =>
    let later := minimum ok rs
    if ok r then some ((later.getD r.energyMJ).min r.energyMJ) else later

/-- Failed quality thresholds never count as zero-cost success. -/
theorem noneIff (ok : Checkpoint → Bool) (rs : List Checkpoint) :
    minimum ok rs = none ↔ ∀ r ∈ rs, ok r = false := by
  induction rs with
  | nil => simp [minimum]
  | cons r rs ih =>
    cases h : ok r <;> simp [minimum, h, ih]

/-- The certified cost is attained and bounds *every* eligible measured checkpoint,
including nonmonotone learning curves and checkpoints listed out of cost order. -/
theorem minimumSpec (ok : Checkpoint → Bool) (rs : List Checkpoint) (m : Nat)
    (hm : minimum ok rs = some m) :
    (∃ r ∈ rs, ok r = true ∧ r.energyMJ = m) ∧
    ∀ r ∈ rs, ok r = true → m ≤ r.energyMJ := by
  induction rs generalizing m with
  | nil => simp [minimum] at hm
  | cons r rs ih =>
    cases h : ok r with
    | false =>
      simp [minimum, h] at hm
      obtain ⟨⟨x,hx,ho,he⟩, lower⟩ := ih m hm
      constructor
      · exact ⟨x,List.mem_cons_of_mem _ hx,ho,he⟩
      · intro y hy good
        rcases List.mem_cons.mp hy with eq | mem
        · subst y; simp [h] at good
        · exact lower y mem good
    | true =>
      cases ht : minimum ok rs with
      | none =>
        simp [minimum,h,ht] at hm
        subst m
        refine ⟨⟨r,by simp,h,rfl⟩,?_⟩
        intro y hy good
        rcases List.mem_cons.mp hy with eq | mem
        · simp [eq]
        · have bad := (noneIff ok rs).mp ht y mem
          simp [bad] at good
      | some n =>
        simp [minimum,h,ht] at hm
        subst m
        obtain ⟨⟨x,hx,ho,he⟩, lower⟩ := ih n ht
        constructor
        · by_cases small : n ≤ r.energyMJ
          · exact ⟨x,List.mem_cons_of_mem _ hx,ho,by simpa [Nat.min_eq_left small] using he⟩
          · exact ⟨r,by simp,h,by simp [Nat.min_eq_right (Nat.le_of_not_ge small)]⟩
        · intro y hy good
          rcases List.mem_cons.mp hy with eq | mem
          · subst y; exact Nat.min_le_right _ _
          · exact Nat.le_trans (Nat.min_le_left _ _) (lower y mem good)

/-- A factor advantage exists on this grid exactly when a qualifying visual
checkpoint beats every qualifying comparator at that factor. Both arms must qualify. -/
theorem factorIff (ok : Checkpoint → Bool) (vis tok : List Checkpoint) (v t factor : Nat)
    (hv : minimum ok vis = some v) (ht : minimum ok tok = some t) :
    factor * v ≤ t ↔
    ∃ x ∈ vis, ok x = true ∧ ∀ y ∈ tok, ok y = true → factor * x.energyMJ ≤ y.energyMJ := by
  obtain ⟨⟨x,hx,goodX,ex⟩, lowestV⟩ := minimumSpec ok vis v hv
  obtain ⟨⟨y,hy,goodY,ey⟩, lowestT⟩ := minimumSpec ok tok t ht
  constructor
  · intro budget
    exact ⟨x,hx,goodX,fun z hz good => by rw [ex]; exact Nat.le_trans budget (lowestT z hz good)⟩
  · rintro ⟨z,hz,good,beats⟩
    exact Nat.le_trans (Nat.mul_le_mul_left factor (lowestV z hz good)) (ey ▸ beats y hy goodY)

end MatchedReadingCost
