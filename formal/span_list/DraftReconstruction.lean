import Std

namespace DraftReconstruction

/-- A and B record complete-meaning credit for two stopped output images.
    pickA is the preference assigned by the reverse translator, ties included. -/
structure Case where
  a : Bool
  b : Bool
  pickA : Bool

def selected (r : Case) := if r.pickA then r.a else r.b
def possible (r : Case) := r.a || r.b
def routingMiss (r : Case) := possible r && !(selected r)
def count (p : Case → Bool) (rows : List Case) := (rows.filter p).length

theorem count_cons (p : Case → Bool) (r : Case) (rs : List Case) :
    count p (r :: rs) = (if p r then 1 else 0) + count p rs := by
  cases h : p r <;> simp [count, h, Nat.add_comm]

/-- Exactly decompose all failures into missing candidates and selection errors. -/
theorem failure_partition (rs : List Case) :
    count selected rs + count routingMiss rs + count (fun r => !possible r) rs = rs.length := by
  induction rs with
  | nil => rfl
  | cons r rs ih =>
    rw [count_cons, count_cons, count_cons, List.length_cons]
    rcases r with ⟨a,b,k⟩
    cases a <;> cases b <;> cases k <;> simp [selected, routingMiss, possible] at ih ⊢ <;> omega

/-- Any source-dependent router choosing one of these two images has this ceiling.
    It is attained by the oracle which selects A precisely when A has credit. -/
theorem router_ceiling (pick : Case → Bool) (rs : List Case) :
    count (fun r => if pick r then r.a else r.b) rs ≤ count possible rs := by
  induction rs with
  | nil => exact Nat.le_refl 0
  | cons r rs ih =>
    rw [count_cons, count_cons]
    cases h : pick r <;> cases ha : r.a <;> cases hb : r.b <;>
      simp [h, ha, hb, possible] <;> omega

theorem attainable (rs : List Case) :
    count (fun r => if r.a then r.a else r.b) rs = count possible rs := by
  congr 1
  funext r
  cases ha : r.a <;> simp [possible,ha]

#print axioms failure_partition
#print axioms router_ceiling
#print axioms attainable
end DraftReconstruction
