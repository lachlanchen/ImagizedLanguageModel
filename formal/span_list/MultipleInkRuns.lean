import Std
namespace MultipleInkRuns

/-- A suffix may skip this cell, begin a new run if budget remains, or extend
a run that was already active. At the end its remaining budget determines the
learned count bonus. All scores are signed fixed-point integers. -/
structure Cell where
  skip : Int
  beginRun : Int
  extendRun : Int

def paths (finish : Nat → Int) : List Cell → Bool → Nat → List Int
  | [], _, k => [finish k]
  | c :: cs, a, k =>
    (paths finish cs false k).map (c.skip + ·) ++
    (match k with
     | 0 => []
     | n+1 => (paths finish cs true n).map (c.beginRun + ·)) ++
    (if a then (paths finish cs true k).map (c.extendRun + ·) else [])

def maximum (x : Int) (xs : List Int) := xs.foldl max x

/-- Attainment and dominance of the best legal complete path using only a
bounded number of run openings, with a question-dependent terminal reward. -/
def optimum (finish : Nat → Int) : List Cell → Bool → Nat → Int
  | [], _, k => finish k
  | c :: cs, a, k =>
    let s := c.skip + optimum finish cs false k
    let b := match k with
      | 0 => s
      | n+1 => max s (c.beginRun + optimum finish cs true n)
    if a then max b (c.extendRun + optimum finish cs true k) else b

theorem exact (f : Nat → Int) (cs : List Cell) (a : Bool) (k : Nat) :
    (optimum f cs a k ∈ paths f cs a k) ∧
    ∀ v ∈ paths f cs a k, v ≤ optimum f cs a k := by
  induction cs generalizing a k with
  | nil => simp [paths, optimum]
  | cons c cs ih =>
    have join {x y : Int} {xs ys : List Int}
        (hx : x ∈ xs ∧ ∀ z ∈ xs, z ≤ x)
        (hy : y ∈ ys ∧ ∀ z ∈ ys, z ≤ y) :
        max x y ∈ xs ++ ys ∧ ∀ z ∈ xs ++ ys, z ≤ max x y := by
      constructor
      · by_cases h : x ≤ y
        · simp [Int.max_eq_right h, hy.1]
        · simp [Int.max_eq_left (by omega : y ≤ x), hx.1]
      · intro z hz
        rcases List.mem_append.mp hz with h|h
        · exact Int.le_trans (hx.2 z h) (Int.le_max_left ..)
        · exact Int.le_trans (hy.2 z h) (Int.le_max_right ..)
    have shift (w : Int) (b : Bool) (n : Nat) :
        w + optimum f cs b n ∈ (paths f cs b n).map (w + ·) ∧
        ∀ z ∈ (paths f cs b n).map (w + ·), z ≤ w + optimum f cs b n := by
      constructor
      · exact List.mem_map.mpr ⟨_, (ih b n).1, rfl⟩
      · intro z hz
        obtain ⟨v,h,rfl⟩ := List.mem_map.mp hz
        exact Int.add_le_add_left ((ih b n).2 v h) _
    cases k with
    | zero =>
      cases a
      · simpa [paths, optimum] using shift c.skip false 0
      · simpa [paths, optimum] using join (shift c.skip false 0) (shift c.extendRun true 0)
    | succ n =>
      have h := join (shift c.skip false (n+1)) (shift c.beginRun true n)
      cases a
      · simpa [paths, optimum] using h
      · simpa [paths, optimum] using join h (shift c.extendRun true (n+1))
#print axioms exact
end MultipleInkRuns
