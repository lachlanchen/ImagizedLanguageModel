import Std

namespace PicturedActionSelection
/-- A strict comparison can be shared across rows. Reversing the physical page
changes the index but preserves the chosen source when the scores differ. -/
def first (left right : Int) : Bool := decide (right ≤ left)

theorem reverse_at_unequal (a b : Int) (h : a ≠ b) :
    first a b = !(first b a) := by
  unfold first
  by_cases ab : a ≤ b <;> by_cases ba : b ≤ a <;> simp_all <;> omega

/-- A question-independent ordering cannot give both desired answers when
the two questions require opposite rows of the same page. -/
theorem need_content {α : Type} (x y : α) (h : x ≠ y) (read : Bool → α) (fixed : Bool) :
    ¬ (read fixed = x ∧ read fixed = y) := by
  rintro ⟨hx, hy⟩
  exact h (hx.symm.trans hy)

/-- For fixed-point scores, every wrongly ordered or tied question incurs
at least one margin of pairwise hinge loss. Summing yields a deterministic
certificate for these comparisons on the set where scores are measured. -/
def hinge (m correct distractor : Int) : Int := max 0 (m + distractor - correct)

def mistakes (xs : List (Int × Int)) : Int :=
  (xs.filter (fun p => decide (p.1 ≤ p.2))).length

def totalHinge (m : Int) : List (Int × Int) → Int
  | [] => 0
  | (a,b) :: xs => hinge m a b + totalHinge m xs

theorem count_bound (xs : List (Int × Int)) (m : Int) :
    mistakes xs * m ≤ totalHinge m xs := by
  induction xs with
  | nil => simp [mistakes, totalHinge]
  | cons a rest ih =>
    by_cases h : a.1 ≤ a.2
    · have hg : m ≤ hinge m a.1 a.2 := by unfold hinge; omega
      simp [mistakes, totalHinge, h, List.filter_cons, Int.add_mul] at ih ⊢
      omega
    · have hg : 0 ≤ hinge m a.1 a.2 := by unfold hinge; omega
      simp [mistakes, totalHinge, h, List.filter_cons] at ih ⊢
      omega

/-- With a single positive, assigning it over half the total positive
weight makes it outrank each distractor. Here the three weights include
the remaining batch entries in `rest`. The normalization cancels. -/
theorem singleton_majority (good bad rest : Nat)
    (h : good + bad + rest < good + good) : bad < good := by omega

/-- The multi-positive training criterion sums positive weights. A large
sum alone cannot certify that *each* requested source beats a distractor. -/
theorem pooled_majority_can_miss : ∃ a b neg : Nat,
    a + b > neg ∧ a < neg ∧ b > neg := by
  exact ⟨1, 6, 5, by decide⟩

/-- If a paraphrase moves both row scores by at most epsilon, an original
strict margin above twice epsilon preserves the source. This condition must
be checked for each action and question frame to certify a paired page. -/
theorem transfer_with_margin (good bad good' bad' eps : Int)
    (margin : bad + 2*eps < good) (lower : good - eps ≤ good')
    (upper : bad' ≤ bad + eps) : bad' < good' := by omega

#print axioms transfer_with_margin
#print axioms singleton_majority
#print axioms pooled_majority_can_miss
#print axioms count_bound
#print axioms reverse_at_unequal
#print axioms need_content
end PicturedActionSelection
