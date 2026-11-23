import Std

namespace ParticipantImageChoice

/-- Two occurrences of one response are disjoint choices of source span.
Masses are nonnegative and can be represented in common fixed-point units. -/
def imageMass (occurrences : List Nat) := occurrences.sum

theorem marginal_may_win : imageMass [4, 4] > imageMass [7] := by decide

/-- Mass borne by other strings can defeat both candidates even when their
restricted ordering identifies the correct person. -/
theorem a_third_image_can_win (correct rival : Nat) (h : rival < correct) :
    ∃ third, rival < correct ∧ correct < third :=
  ⟨correct + 1, h, by omega⟩

/-- Checking strict choices in both orders treats an unresolved tie as a miss. -/
theorem both_questions (a b c d : Int) :
    (a > b ∧ d > c) ↔ (a - b > 0 ∧ c - d < 0) := by omega

#print axioms both_questions
#print axioms a_third_image_can_win
end ParticipantImageChoice
