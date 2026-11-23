import Std
namespace QuestionAssignment

/-- Row- and column-only preferences cancel in the two possible assignments.
Any strict win must use a cross-question interaction. Scores use signed units. -/
theorem cancel_nuisance (s t u v a b x y : Int) :
    (s+a+x) + (v+b+y) - ((t+a+y) + (u+b+x)) = s+v-(t+u) := by omega

/-- Stable assignment when the correct total exceeds the rival by more than
four times the largest error in an individual compatibility score. -/
theorem stable (s t u v ds dt du dv e : Int)
    (h : 4*e < s+v-(t+u))
    (hs : -e ≤ ds) (hv : -e ≤ dv) (ht : dt ≤ e) (hu : du ≤ e) :
    (t+dt)+(u+du) < (s+ds)+(v+dv) := by omega

#print axioms cancel_nuisance
#print axioms stable
end QuestionAssignment
