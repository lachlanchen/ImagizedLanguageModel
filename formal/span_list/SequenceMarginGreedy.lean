import Std

namespace SequenceMarginGreedy

/-- The first wrong cell in a fixed reference path has a rival at least
as plausible as the reference's cell. Strictly positive local margins at
every prefix, including the final stop, suffice for greedy reproduction. -/
theorem greedy_path_of_local_margins
    (choose : List α → α) (score : List α → α → Int)
    (best : ∀ history other, score history other ≤ score history (choose history))
    (history continuation : List α)
    (wins : ∀ (before : List α) (cell : α) (after : List α),
      continuation = before ++ cell :: after → ∀ x, x ≠ cell →
      score (history ++ before) x < score (history ++ before) cell) :
    ∀ (before : List α) (cell : α) (after : List α),
    continuation = before ++ cell :: after → choose (history ++ before) = cell := by
  intro before cell after split
  by_cases h : choose (history ++ before) = cell
  · exact h
  · have lower := best (history ++ before) cell
    have upper := wins before cell after split _ h
    omega

def emit (choose : List α → α) (history : List α) : Nat → List α
  | 0 => []
  | n+1 => let cell := choose history
           cell :: emit choose (history ++ [cell]) n

theorem exact_emission (choose : List α → α) (history target : List α)
    (correct : ∀ (p : List α) (c : α) (q : List α),
       target = p ++ c :: q → choose (history ++ p) = c) :
    emit choose history target.length = target := by
  induction target generalizing history with
  | nil => rfl
  | cons a rest ih =>
    have head : choose history = a := by simpa using correct [] a rest rfl
    simp only [List.length_cons, emit, head, List.cons.injEq, true_and]
    apply ih
    intro p c q eqn
    have same : a :: rest = (a :: p) ++ c :: q := by simp [eqn]
    simpa [List.append_assoc] using correct (a :: p) c q same

/-- Greedy can miss even the unique globally likeliest equal-length string.
Root mass 4/10 goes to the correct branch, where the continuation is certain.
The rival 6/10 splits 6:4. Joint path masses have common denominator 100.
Thus ranking the correct complete path above every alternative does not
ensure that the normal incremental decoder will produce it. -/
theorem full_path_order_does_not_fix_first_choice :
    (4:Nat)*10 > 6*6 ∧ (4:Nat)*10 > 6*4 ∧ (6:Nat)>4 := by decide

/-- Cross multiplying the length-normalized preference gap by the two
positive lengths isolates the exact coefficient on a common history.
When the correct answer is longer, increasing a shared prefix score
reduces the ranking gap; equal lengths cancel the shared score exactly.
The formula applies to fixed-point log scores at any common scale. -/
theorem shared_prefix_coefficient (goodLen badLen common goodTail badTail : Int) :
    badLen * (common + goodTail) - goodLen * (common + badTail) =
    (badLen - goodLen) * common + badLen * goodTail - goodLen * badTail := by
  simp only [Int.mul_add, Int.sub_mul]
  omega

end SequenceMarginGreedy
