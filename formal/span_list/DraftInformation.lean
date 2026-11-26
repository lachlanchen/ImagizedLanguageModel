import Std

namespace DraftInformation

/-- A deterministic phrase transducer makes a second image from the first. -/
def alongDraft (draft : Source → Draft) (edit : Source → Draft → Answer) : Source → Answer :=
  fun s => edit s (draft s)

/-- Any paired source-and-draft decision rule has an observationally
equivalent source rule on every possible input; conversely the editor can
ignore its draft. The equivalence survives any finite panel and scoring rule. -/
theorem attainable_scores (draft : Source → Draft) (panel : List Source)
    (score : Source → Answer → Nat) (marks : List Nat) :
    (∃ edit : Source → Draft → Answer,
       panel.map (fun s => score s (edit s (draft s))) = marks) ↔
    (∃ read : Source → Answer,
       panel.map (fun s => score s (read s)) = marks) := by
  constructor
  · rintro ⟨edit, h⟩
    exact ⟨alongDraft draft edit, h⟩
  · rintro ⟨read, h⟩
    exact ⟨fun s _ => read s, h⟩

/-- A reader that loses the original image incurs at least one error whenever
two collapsed drafts require incompatible answers. This uses an arbitrary
acceptance relation, so multiple admissible translations are allowed. -/
theorem incompatible_drafts (draft : Source → Draft) (accept : Source → Answer → Prop)
    (a b : Source) (same : draft a = draft b)
    (conflict : ∀ answer, accept a answer → ¬ accept b answer) (recover : Draft → Answer) :
    ¬ (accept a (recover (draft a)) ∧ accept b (recover (draft b))) := by
  rintro ⟨ha, hb⟩
  rw [← same] at hb
  exact conflict _ ha hb

end DraftInformation
