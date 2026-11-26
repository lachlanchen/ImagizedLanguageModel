import Std
namespace PhraseCoverage

/-- A completed path partitions source positions: the coverage check inspects
the concatenation of the consumed spans, independently of target spelling. -/
def covered (spans : List (List Nat)) (n : Nat) : Prop :=
  spans.join.Perm (List.range n)

theorem complete_budget (spans : List (List Nat)) (n : Nat)
    (ok : covered spans n) : spans.join.length = n := by
  have h := ok.length_eq
  simpa using h

theorem consumes_exactly_once (spans : List (List Nat)) (n i : Nat)
    (ok : covered spans n) : spans.join.Nodup ∧ (i ∈ spans.join ↔ i < n) := by
  constructor
  · exact ok.nodup_iff.mpr (List.nodup_range n)
  · exact ok.mem_iff.trans List.mem_range

/-- A model may satisfy coverage while emitting any one target value per span. -/
theorem emissions_unconstrained (spans : List (List Nat)) (n : Nat)
    (ok : covered spans n) (values : List α) (h : values.length = spans.length) :
    covered ((spans.zip values).map Prod.fst) n := by
  rw [List.map_fst_zip spans values (by omega)]
  exact ok

end PhraseCoverage
