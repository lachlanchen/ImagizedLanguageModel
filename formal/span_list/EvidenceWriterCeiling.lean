import Std

namespace EvidenceWriterCeiling

/-- Each entry specifies the frozen writer's outcome under one legal evidence
strip. The selected strip is explicitly a member, including when strips tie. -/
structure Observation where
  chosen : Bool
  alternatives : List Bool
  admissible : chosen ∈ alternatives

def reachable (o : Observation) : Bool := o.alternatives.any id

def errors (rows : List Observation) : Nat := (rows.filter fun o => !o.chosen).length
def unreachable (rows : List Observation) : Nat := (rows.filter fun o => !reachable o).length
def missed (rows : List Observation) : Nat :=
  (rows.filter fun o => reachable o && !o.chosen).length

theorem chosen_reachable (o : Observation) (h : o.chosen = true) : reachable o = true := by
  apply List.any_eq_true.mpr
  exact ⟨o.chosen, o.admissible, h⟩

/-- The first term is invariant under any change to the selector that leaves
the writer, candidate set, decoder and acceptance predicate fixed. -/
theorem decompose (rows : List Observation) :
    errors rows = unreachable rows + missed rows := by
  induction rows with
  | nil => rfl
  | cons o tail ih =>
    have h := chosen_reachable o
    cases a : o.chosen <;> cases b : reachable o <;>
      simp [errors, unreachable, missed, a, b] at ih ⊢ <;> first | omega | simp_all

end EvidenceWriterCeiling
