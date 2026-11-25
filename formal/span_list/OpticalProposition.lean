import Std

namespace OpticalProposition

/-- A supplied common prefix, the discriminating glyph and the remaining
    sentence are separate observable events in the free run. -/
structure Trial where
  prefixOK : Bool
  branchOK : Bool
  suffixAndStopOK : Bool
  deriving DecidableEq

def correct (x : Trial) : Bool := x.prefixOK && x.branchOK && x.suffixAndStopOK

theorem full_requires_free_prefix (x : Trial) (h : correct x = true) :
    x.prefixOK = true ∧ x.branchOK = true := by
  cases x with | mk p b s => cases p <;> cases b <;> cases s <;> simp_all [correct]

theorem counts (xs : List Trial) :
    (xs.filter correct).length ≤ (xs.filter Trial.prefixOK).length ∧
    (xs.filter correct).length ≤ (xs.filter Trial.branchOK).length := by
  induction xs with
  | nil => simp
  | cons a rest ih =>
    rcases ih with ⟨hp, hb⟩
    cases a with | mk p b s =>
      cases p <;> cases b <;> cases s <;> simp_all [List.filter, correct] <;> omega

/-- A single source-independent preference at the identical prefix cannot
    strictly prefer each distinct next glyph over the other. -/
theorem source_independent_fails (a b : Int) : ¬(a > b ∧ b > a) := by omega

/-- For equal-length whole-reply candidates, a reverse score can overturn
    a wrong forward ordering exactly when it offsets the forward advantage.
    Scores may also be length-normalised before fixed-point conversion. -/
theorem reverse_rescue (fGood fBad rGood rBad : Int) :
    (fGood + rGood > fBad + rBad) ↔ rGood-rBad > fBad-fGood := by omega

end OpticalProposition

namespace OpticalProposition

/-- Integer similarity units, e.g. cosine rounded to millionths. If each
    measured score changes by at most e, a strict gap exceeding 2e certifies
    the same top-ranked translation among all alternatives. -/
theorem stable_match {ι : Type} (gold : ι) (s p : ι → Int) (e : Int)
    (lo : ∀ i, s i - e ≤ p i) (hi : ∀ i, p i ≤ s i + e)
    (gap : ∀ i, i ≠ gold → 2 * e < s gold - s i) :
    ∀ i, i ≠ gold → p i < p gold := by
  intro i h
  have a := lo gold
  have b := hi i
  have c := gap i h
  omega

/-- Across requests, returning only candidates from each proposed list
    limits the number of exact answers any subsequent selector can recover. -/
theorem selection_coverage {α : Type} [DecidableEq α]
    (cases : List (α × List α × α))
    (valid : ∀ x ∈ cases, x.2.2 ∈ x.2.1) :
    (cases.filter (fun x => decide (x.1 = x.2.2))).length ≤
    (cases.filter (fun x => decide (x.1 ∈ x.2.1))).length := by
  induction cases with
  | nil => simp
  | cons x tail ih =>
    have mem := valid x (by simp)
    have bound := ih (fun a h => valid a (by simp [h]))
    by_cases eq : x.1 = x.2.2
    · have inc : x.1 ∈ x.2.1 := eq ▸ mem
      simp [List.filter, eq, mem] at * <;> omega
    · by_cases inc : x.1 ∈ x.2.1 <;> simp [List.filter, eq, inc] <;> omega

end OpticalProposition
