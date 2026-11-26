import Std

namespace SupportedWritingMargin

/-- Scores may be expressed in any common integer unit. A support head
supplies a nonnegative bounded first-emission bonus to each competitor.
The required margin includes the bonus paid to the competing action. -/
theorem strict_repair_iff (gold rival gBonus rBonus : Int) :
    rival + rBonus < gold + gBonus ↔ rival - gold < gBonus - rBonus := by
  omega

theorem unreachable (gold rival gBonus rBonus cap : Int)
    (gap : cap < rival - gold) (bounded : gBonus ≤ cap) (nonneg : 0 ≤ rBonus) :
    gold + gBonus < rival + rBonus := by omega

/-- For a whole supplied path, even a best possible bounded head must miss
every step whose original best competitor exceeds the cap. This includes
the stop decision if the original writer fails to end the answer. -/
theorem compulsory_errors (cases : List (Int × Int)) (cap : Int) (wins : Int × Int → Bool)
    (sound : ∀ p ∈ cases, cap < p.2 - p.1 → wins p = false) :
    (cases.filter (fun p => decide (cap < p.2 - p.1))).length ≤
    (cases.filter (fun p => !(wins p))).length := by
  induction cases with
  | nil => simp
  | cons a tail ih =>
    have h := ih (fun p hp => sound p (List.mem_cons_of_mem _ hp))
    by_cases bad : cap < a.2 - a.1
    · have hw := sound a (by simp) bad
      simp [List.filter, bad, hw, h]
    · cases hw : wins a <;> simp [List.filter, bad, hw] <;> omega

end SupportedWritingMargin
