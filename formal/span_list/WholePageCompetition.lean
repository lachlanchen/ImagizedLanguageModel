import Std
namespace WholePageCompetition

/-- Finite paths may have repeated source placements for an answer. The
chosen positive need only be admissible for the loss to bound an actual decoding
mistake. Comparing different page lengths requires averaging per page. -/
theorem free_mistake_bound {α : Type} (score : α → Int) (accepted : α → Bool)
    (free gold : α) (optimal : ∀ a, score a ≤ score free) (unit : Int) (hu : 0 < unit) :
    (if accepted free then 0 else unit) ≤
       (if accepted free then 0 else max 0 (unit + score free - score gold)) := by
  have h := optimal gold
  cases accepted free <;> simp <;> omega

def total : List Int → Int
  | [] => 0
  | x :: xs => x + total xs

/-- A branch's suffix bound can ignore mutual overlap: replacing any
remaining positional choice by its individual maximum only increases the sum. -/
theorem sum_upper (xs : List (Int × Int)) (h : ∀ z ∈ xs, z.1 ≤ z.2) :
    total (xs.map Prod.fst) ≤ total (xs.map Prod.snd) := by
  induction xs with
  | nil => simp [total]
  | cons a t ih =>
    have ha := h a (by simp)
    have ht := ih (fun z hz => h z (by simp [hz]))
    simp only [List.map_cons, total]
    omega

theorem prune (partialScore actual bound incumbent : Int) (ha : actual ≤ bound)
    (hb : partialScore + bound ≤ incumbent) : partialScore + actual ≤ incumbent := by omega
#print axioms free_mistake_bound
end WholePageCompetition
