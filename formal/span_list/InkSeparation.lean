import Std

namespace InkSeparation

/-- Bit-distance certificate for an entire corruption neighbourhood.
    sep is the minimum number of differing pixels from prototype c to any
    other allowed prototype. e bounds further flips from the observed q. -/
theorem stable {α : Type} (d : α → α → Nat)
    (triangle : ∀ x y z, d x z ≤ d x y + d y z)
    (symm : ∀ x y, d x y = d y x)
    (bank : List α) (c q q' : α) (sep e : Nat)
    (separated : ∀ b ∈ bank, b ≠ c → sep ≤ d c b)
    (close : d q' q ≤ e) (margin : 2 * (d q c + e) < sep) :
    2 * d q' c < sep ∧ ∀ b ∈ bank, b ≠ c → d q' c < d q' b := by
  have near := triangle q' q c
  constructor
  · omega
  · intro b hb hbc
    have h := separated b hb hbc
    have t := triangle c q' b
    rw [symm c q'] at t
    omega

/-- Each position is a pixel in the 1,024-pixel measurement window. -/
def ham {α : Type} (a b : α → Bool) : List α → Nat
  | [] => 0
  | x :: xs => (if a x = b x then 0 else 1) + ham a b xs

theorem ham_triangle {α : Type} (a b c : α → Bool) (xs : List α) :
    ham a c xs ≤ ham a b xs + ham b c xs := by
  induction xs with
  | nil => simp [ham]
  | cons x xs ih =>
    simp only [ham]
    have single : (if a x = c x then 0 else 1 : Nat) ≤
        (if a x = b x then 0 else 1) + (if b x = c x then 0 else 1) := by
      cases a x <;> cases b x <;> cases c x <;> decide
    omega

theorem ham_symm {α : Type} (a b : α → Bool) (xs : List α) :
    ham a b xs = ham b a xs := by
  induction xs with
  | nil => rfl
  | cons x xs ih =>
    simp only [ham, ih]
    cases a x <;> cases b x <;> rfl

theorem bitmap_stable {α : Type} (positions : List α) (bank : List (α → Bool))
    (c q q' : α → Bool) (sep e : Nat)
    (separated : ∀ b ∈ bank, b ≠ c → sep ≤ ham c b positions)
    (close : ham q' q positions ≤ e) (margin : 2 * (ham q c positions + e) < sep) :
    2 * ham q' c positions < sep ∧
    ∀ b ∈ bank, b ≠ c → ham q' c positions < ham q' b positions :=
  stable (fun a b => ham a b positions) (fun a b c => ham_triangle a b c positions)
    (fun a b => ham_symm a b positions) bank c q q' sep e separated close margin

/-- For any fixed rubric, the gain from a new reader is bounded by the number
    of fully emitted answers that change. The input can include stop and pixels. -/
def total : List Nat → Nat
  | [] => 0
  | x :: xs => x + total xs

theorem score_gain_bound {α : Type} [DecidableEq α]
    (items : List (α × α × (α → Bool))) :
    total (items.map (fun (_,b,r) => (r b).toNat)) ≤
    total (items.map (fun (a,_,r) => (r a).toNat)) +
    total (items.map (fun (a,b,_) => if a = b then 0 else 1)) := by
  induction items with
  | nil => simp [total]
  | cons h t ih =>
    rcases h with ⟨a,b,r⟩
    simp only [List.map_cons, total]
    have one : (r b).toNat ≤ (r a).toNat + (if a = b then 0 else 1) := by
      by_cases same : a = b
      · subst b
        simp
      · simp only [if_neg same]
        cases r a <;> cases r b <;> decide
    exact by simpa only [Nat.add_assoc, Nat.add_left_comm, Nat.add_comm] using Nat.add_le_add one ih

end InkSeparation
