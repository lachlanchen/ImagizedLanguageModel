import Std
namespace QuestionedEvents

/-- For a fixed verb its question score is constant. An object that maximizes
the arc score thus also maximizes the conditional joint score for that verb. -/
theorem object_max {V O : Type} (a : V → O → Int) (q : V → Int)
    (v : V) (o : O) (best : ∀ j, a v j ≤ a v o) :
    ∀ j, a v j + q v ≤ a v o + q v := by
  intro j
  have h := best j
  omega

/-- Selecting each verb's best object and comparing those maxima returns a
global maximizer of the joint score. Ties allow several valid maximizers. -/
theorem joint_max {V O : Type} (a : V → O → Int) (q : V → Int)
    (choose : V → O) (inner : ∀ v o, a v o ≤ a v (choose v))
    (u : V) (outer : ∀ v, a v (choose v) + q v ≤ a u (choose u) + q u) :
    ∀ v o, a v o + q v ≤ a u (choose u) + q u := by
  intro v o
  exact Int.le_trans (object_max a q v (choose v) (inner v) o) (outer v)

#print axioms joint_max
end QuestionedEvents
