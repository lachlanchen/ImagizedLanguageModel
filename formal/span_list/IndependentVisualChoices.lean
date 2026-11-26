import Std
namespace IndependentVisualChoices

/-- u and v are the margins of the true answer on each source, in opposite
candidate order. Remove the factor of two to use exact score units. -/
theorem both_iff_signal_dominates (u v : Int) :
    (0 < u ∧ 0 < v) ↔ max (u-v) (v-u) < u+v := by
  rw [Int.max_def]
  split <;> omega

/-- A source-independent preference correction shifts the two margins in
opposite directions and preserves the joint assignment. -/
theorem common_prior_cancels (u v prior : Int) :
    (u - prior) + (v + prior) = u + v := by omega

/-- Characterise every common correction that gives both sources their
correct answer. The interval can be empty even with a fixed correct label. -/
theorem viable_correction (u v c : Int) :
    (0 < u - c ∧ 0 < v + c) ↔ -v < c ∧ c < u := by omega

end IndependentVisualChoices
