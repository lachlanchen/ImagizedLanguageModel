import Std

namespace CompetingFacts

/-- Two questions share a page. a and b are the first question's complete
interval log scores for its own and rival bitmap; c and d are the corresponding
rival and own scores for the second question. One page-specific bias t favours
the first bitmap equally on both questions. Doubling allows half-unit offsets
while keeping exact, signed fixed-point scores. -/
theorem simultaneous_margin (a b c d m : Int) :
    (∃ t : Int, 2 * (a - b) + t > 2*m ∧ 2 * (d - c) - t > 2*m)
    ↔ (a - b) + (d - c) > 2*m := by
  constructor
  · rintro ⟨t,h,k⟩
    omega
  · intro h
    exact ⟨(d-c)-(a-b), by omega, by omega⟩

/-- The smaller margin can never exceed the bias-invariant interaction.
The midpoint bias achieves this bound in both directions simultaneously. -/
theorem optimal_worst_margin (a b c d : Int) :
    (∀ t : Int, min (2*(a-b)+t) (2*(d-c)-t) ≤ (a-b)+(d-c)) ∧
    (∃ t : Int, 2*(a-b)+t = (a-b)+(d-c) ∧
               2*(d-c)-t = (a-b)+(d-c)) := by
  constructor
  · intro t
    by_cases h : 2*(a-b)+t ≤ 2*(d-c)-t
    · rw [Int.min_eq_left h]; omega
    · rw [Int.min_eq_right (by omega)]; omega
  · exact ⟨(d-c)-(a-b), by omega, by omega⟩
#print axioms simultaneous_margin
#print axioms optimal_worst_margin
/-- For four valid intervals of the same question and page, remove the
learned length score before evaluating this identity. The normalization
constant also cancels. A separable endpoint score has no residual interaction. -/
theorem endpoint_rectangle (u v x y : Int) :
    (u + x) + (v + y) - (u + y) - (v + x) = 0 := by omega

/-- After length correction, the two start positions cannot induce opposite
strict preferences between the same two ends. A rank-one product can. -/
theorem no_crossed_preferences (u v x y : Int) :
    ¬ (u + x > u + y ∧ v + y > v + x) := by omega

example : (1 : Int) * 1 > 1 * (-1) ∧ (-1 : Int) * (-1) > (-1) * 1 := by decide

#print axioms endpoint_rectangle
#print axioms no_crossed_preferences
end CompetingFacts
