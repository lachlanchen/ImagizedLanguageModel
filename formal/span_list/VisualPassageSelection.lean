import Std

namespace VisualPassageSelection

/-- Scaled copy-event masses at every location whose physical spelling matches. -/
def deposited [DecidableEq α] (image : α) : List (α × Nat) → Nat
  | [] => 0
  | (x, a) :: xs => (if x = image then a else 0) + deposited image xs

/-- A repeated printed word may have several distinguishable reading locations.
Removing their locations before comparing whole-image masses preserves the sum. -/
theorem concatenate [DecidableEq α] (w : α) (a b : List (α × Nat)) :
    deposited w (a ++ b) = deposited w a + deposited w b := by
  induction a with
  | nil => simp [deposited]
  | cons h t ih => cases h; simp [deposited, ih, Nat.add_assoc]

theorem absent [DecidableEq α] (w : α) (xs : List (α × Nat))
    (h : ∀ p ∈ xs, p.1 ≠ w) : deposited w xs = 0 := by
  induction xs with
  | nil => rfl
  | cons p t ih =>
    obtain ⟨x, a⟩ := p
    have ne : x ≠ w := h _ (List.mem_cons_self ..)
    have ht : ∀ p ∈ t, p.1 ≠ w := fun p hp => h p (List.mem_cons_of_mem _ hp)
    simp [deposited, ne, ih ht]

/-- For a pictured winner w and a rival v this is the exact condition on the
pointer advantage needed to overturn the spelling head's contrary preference.
The components include their mixture weights and share a common denominator. -/
theorem crossRouteMargin [DecidableEq α] (w v : α) (xs : List (α × Nat))
    (spell : α → Nat) (prefers : spell w ≤ spell v) :
    spell v + deposited v xs < spell w + deposited w xs ↔
    deposited v xs + (spell v - spell w) < deposited w xs := by omega

/-- All visible, printable page candidates were enumerated. Any outside rival
must win by spelling alone. A bound on the remaining spelling search therefore
certifies a global decision without enumerating every alphabetic string. -/
theorem searchCertificate [DecidableEq α] (winner : α) (xs : List (α × Nat))
    (spell : α → Nat) (listed : α → Prop) (upper : Nat)
    (covers : ∀ p ∈ xs, listed p.1)
    (best : ∀ x, listed x → spell x + deposited x xs ≤ spell winner + deposited winner xs)
    (bound : ∀ x, ¬listed x → spell x ≤ upper)
    (margin : upper ≤ spell winner + deposited winner xs) :
    ∀ x, spell x + deposited x xs ≤ spell winner + deposited winner xs := by
  intro x
  by_cases hx : listed x
  · exact best x hx
  · have hz := absent x xs (fun p hp eq => hx (eq ▸ covers p hp))
    simpa [hz] using Nat.le_trans (bound x hx) margin

end VisualPassageSelection
