import Std
namespace InkRunSurvival

/-- Numerators for opening and continuing at a pictured cell. If a run closes,
the same cell is eligible to open a new run. Both binary choices have denominator d. -/
structure Cell where
  opened : Nat
  skipped : Nat
  stayed : Nat
  closed : Nat
  d : Nat
  open_total : skipped + opened = d
  stay_total : closed + stayed = d

def total : List Cell → Bool → Nat
  | [], _ => 1
  | c :: cs, false => c.skipped*c.d*total cs false + c.opened*c.d*total cs true
  | c :: cs, true => c.closed*c.skipped*total cs false +
      c.closed*c.opened*total cs true + c.stayed*c.d*total cs true

def denominator : List Cell → Nat
  | [] => 1
  | c :: cs => (c.d*c.d)*denominator cs

/-- Sum over all legal fact lists, before any cap on count or learned count reward.
For positive denominators division gives exactly one, for arbitrary page length. -/
theorem unit_mass (cs : List Cell) (on : Bool) : total cs on = denominator cs := by
  induction cs generalizing on with
  | nil => cases on <;> rfl
  | cons c cs ih =>
    cases on
    · simp only [total, denominator, ih, ← Nat.add_mul, c.open_total]
    · simp only [total, denominator, ih, ← Nat.add_mul]
      rw [← Nat.mul_add, c.open_total, ← Nat.add_mul, c.stay_total]

/-- A path's exit costs can be charged to the preceding selected cells.
In the following sums the cost at a cell is a signed fixed-point log cost,
and the page end has zero exit cost. -/
def preceding (active : Bool) : List (Bool × Int) → Int
  | [] => 0
  | (next, cost) :: xs => (if active then cost else 0) + preceding next xs

def moved : List (Bool × Int) → Int
  | [] => 0
  | (active,_) :: xs =>
    (if active then (xs.head?.map Prod.snd).getD 0 else 0) + moved xs

theorem shift_exit (xs : List (Bool × Int)) (active : Bool) :
    preceding active xs = (if active then (xs.head?.map Prod.snd).getD 0 else 0) + moved xs := by
  induction xs generalizing active with
  | nil => cases active <;> rfl
  | cons h t ih =>
    obtain ⟨n,c⟩ := h
    simp [preceding, moved, ih n]
#print axioms unit_mass
#print axioms shift_exit
end InkRunSurvival
