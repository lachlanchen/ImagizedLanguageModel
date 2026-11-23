import Std

namespace ClassicalArgumentMass

/-- A full-page loss and a two-name loss share the per-answer negative log
mass. All terms here are log masses expressed in common fixed-point units. -/
def score (allMass nameMass answerMass : Int) :=
  (allMass - answerMass) + (nameMass - answerMass)

theorem discriminating_part (allMass nameMass a b : Int) :
    score allMass nameMass b - score allMass nameMass a = 2 * (a-b) := by
  unfold score
  omega

/-- The equal-name control is invariant under exchanging the assignment. -/
theorem equal_control (allMass nameMass a b : Int) :
    score allMass nameMass a + score allMass nameMass b =
    score allMass nameMass b + score allMass nameMass a := by omega

/-- Returning a page position can follow an exchange of the two names. Even
four successes cannot by themselves prove that the reader interpreted a verb. -/
def atPosition {α : Type} (a b : α) (second edited : Bool) : α :=
  if second != edited then b else a

theorem positional_witness {α : Type} (a b : α) :
    (atPosition a b false false, atPosition a b true false,
     atPosition a b false true, atPosition a b true true) = (a,b,b,a) := rfl

#print axioms discriminating_part
#print axioms equal_control
#print axioms positional_witness
end ClassicalArgumentMass
