import Std

namespace VisualStripRecovery

/-- Whole-answer exact credit for one deterministic decoder. -/
def credit [DecidableEq α] (written gold : α) : Nat := if written = gold then 1 else 0

/-- If reversing the pictured subject and object preserves the unordered strip
inventory, a writer seeing only that inventory earns at most one exact credit.
This holds for any representation of the inventory and any deterministic decoder. -/
theorem collision_limit [DecidableEq α] (observe : α → β) (write : β → α)
    (x y : α) (distinct : x ≠ y) (same : observe x = observe y) :
    credit (write (observe x)) x + credit (write (observe y)) y ≤ 1 := by
  rw [← same]
  unfold credit
  split <;> split <;> simp_all

def total : List Nat → Nat
  | [] => 0
  | x :: xs => x + total xs

/-- The bound sums over disjoint pair instances without an independence assumption. -/
theorem panel_limit [DecidableEq α] (observe : α → β) (write : β → α)
    (pairs : List (α × α))
    (ambiguous : ∀ p ∈ pairs, p.1 ≠ p.2 ∧ observe p.1 = observe p.2) :
    total (pairs.map (fun p => credit (write (observe p.1)) p.1 +
                          credit (write (observe p.2)) p.2)) ≤ pairs.length := by
  induction pairs with
  | nil => simp [total]
  | cons p ps ih =>
    obtain ⟨hd, hs⟩ := ambiguous p (by simp)
    have h := collision_limit observe write p.1 p.2 hd hs
    have rest := ih (fun a ha => ambiguous a (by simp [ha]))
    simp only [List.map_cons, total, List.length_cons]
    omega

/-- Correct strip length is necessary for exact recovery of the canonical
action sequence. Other segmentations may still emit the same physical sentence. -/
theorem wrong_length_excludes_action_recovery (written gold : List α)
    (h : written.length ≠ gold.length) : written ≠ gold := by
  intro equal
  exact h (congrArg List.length equal)

/-- In a two-position assignment, changing the baseline score of either
position or either strip leaves the preferred orientation unchanged. The
experimental margin can therefore use logits without per-row log normalizers. -/
theorem assignment_offset_invariance (aa ab ba bb r s u v : Int) :
    ((aa+r+u)+(bb+s+v)) - ((ab+r+v)+(ba+s+u)) =
    (aa+bb)-(ab+ba) := by omega

end VisualStripRecovery
