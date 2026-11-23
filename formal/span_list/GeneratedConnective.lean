import Std

namespace GeneratedConnective
variable {G : Type}

/-- Every redirected or repeated source span still draws its cells from the page. -/
theorem copied_chunks_stay_on_page (page : List G) (chunks : List (List G))
    (contained : ∀ s ∈ chunks, ∀ x ∈ s, x ∈ page) (g : G)
    (seen : g ∈ chunks.join) : g ∈ page := by
  obtain ⟨s, hs, hg⟩ := List.mem_join.mp seen
  exact contained s hs g hg

/-- Exact success on a reply containing a new connector witnesses a writing
operation even if a reader may copy any collection of disjoint intervals. -/
theorem necessary_write (page a b : List G) (g : G) (newInk : g ∉ page)
    (chunks : List (List G)) (contained : ∀ s ∈ chunks, ∀ x ∈ s, x ∈ page) :
    chunks.join ≠ a ++ [g] ++ b := by
  intro same
  apply newInk
  apply copied_chunks_stay_on_page page chunks contained
  rw [same]
  simp

#print axioms necessary_write
end GeneratedConnective
