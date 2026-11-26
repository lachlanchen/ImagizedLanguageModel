import Std

namespace PairedInkOdds

/-- Crossed minus matched energy, in fixed-point log-likelihood units. -/
def gap (s00 s01 s10 s11 : Int) : Int := s00 + s11 - s01 - s10

/-- Neither source-only evidence (including a shared target prefix at that
    source) nor target-only fluency can create a pairing advantage. -/
theorem interaction_only (a b c d x0 x1 y0 y1 : Int) :
    gap (a+x0+y0) (b+x0+y1) (c+x1+y0) (d+x1+y1) = gap a b c d := by
  unfold gap
  omega

/-- The score is an interaction. Both row-wise decisions are correct exactly
    when the interaction exceeds the absolute imbalance of their margins. -/
theorem both_rows (a b c d : Int) :
    (a > b ∧ d > c) ↔
    ((a-b)-(d-c) < gap a b c d ∧ (d-c)-(a-b) < gap a b c d) := by
  unfold gap
  omega

/-- Perfect pairing at arbitrarily large positive margin coexists with a
    third full image sequence ranked above both correct replies at each source. -/
theorem third_reply (m : Nat) :
    ∃ a b c d e f : Int, gap a b c d > (m : Int) ∧ a > b ∧ d > c ∧ e > a ∧ f > d := by
  refine ⟨-1, -(m : Int)-2, -(m : Int)-2, -1, 0, 0, ?_⟩
  unfold gap
  omega

def total : List (Bool × Nat) → Nat
  | [] => 0
  | x :: xs => x.2 + total xs

/-- Two-class log loss charges at least one bit to every tied or incorrect
    decision. Once that per-example bound is supplied, it lifts to a count
    certificate at any fixed-point resolution. -/
theorem mistake_budget (scale : Nat) (xs : List (Bool × Nat))
    (h : ∀ p ∈ xs, p.1 = true → scale ≤ p.2) :
    (xs.filter (fun p => p.1)).length * scale ≤ total xs := by
  induction xs with
  | nil => simp [total]
  | cons a t ih =>
    have ht := ih (fun p hp => h p (by simp [hp]))
    have ha := h a (by simp)
    cases a with
    | mk b n =>
      cases b <;> simp_all [total, List.filter, Nat.succ_mul] <;> omega

end PairedInkOdds

namespace PairedInkOdds

/-- If the two pictured sources collapse to one representation, the two
    possible assignments have identical total log score for every decoder. -/
theorem collision_gap {X Y Z : Type} (encode : X → Z) (s : Z → Y → Int)
    (u v : X) (a b : Y) (h : encode u = encode v) :
    gap (s (encode u) a) (s (encode u) b) (s (encode v) a) (s (encode v) b) = 0 := by
  rw [h]
  unfold gap
  omega

theorem collision_blocks_two_replies {X Y Z : Type} (encode : X → Z)
    (draw : Z → Y) (u v : X) (a b : Y) (same : encode u = encode v)
    (different : a ≠ b) : ¬ (draw (encode u) = a ∧ draw (encode v) = b) := by
  rintro ⟨ha,hb⟩
  exact different (ha.symm.trans ((congrArg draw same).trans hb))

end PairedInkOdds
