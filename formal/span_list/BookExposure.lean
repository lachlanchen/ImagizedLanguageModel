import Std

namespace BookExposure

/-- The number of input or target positions presented over a training schedule. -/
def positions {α : Type} (size : α → Nat) : List α → Nat
  | [] => 0
  | x :: xs => size x + positions size xs

/-- Substitution within exact shape strata preserves the exposure in each batch,
    including arbitrary repetitions and either translation direction. -/
theorem matched_positions {α : Type} (size : α → Nat) (replace : α → α)
    (rows : List α) (h : ∀ x, size (replace x) = size x) :
    positions size (rows.map replace) = positions size rows := by
  induction rows with
  | nil => rfl
  | cons a t ih => simp [positions, h, ih]

/-- In a fibre with c distinct broad rows, any of its c presentation slots
    could independently contain any one of c source-target pairs. -/
def possibilities : List Nat → Nat
  | [] => 1
  | c :: cs => c ^ c * possibilities cs

def discardedBits : List Nat → Nat
  | [] => 0
  | c :: cs => c.log2 * c + discardedBits cs

/-- A finite lower bound for the conditional information lost when every
    fibre is replaced by one representative. Zero-count fibres are omitted. -/
theorem ambiguity_bound (counts : List Nat) (positive : ∀ c ∈ counts, c ≠ 0) :
    2 ^ discardedBits counts ≤ possibilities counts := by
  induction counts with
  | nil => simp [discardedBits, possibilities]
  | cons c cs ih =>
    have h := Nat.pow_le_pow_left (Nat.log2_self_le (positive c (by simp))) c
    have tail := ih (fun x hx => positive x (by simp [hx]))
    simpa [discardedBits, possibilities, Nat.pow_add, Nat.pow_mul] using
      (Nat.mul_le_mul h tail)

end BookExposure
