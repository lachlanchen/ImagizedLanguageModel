import Std

namespace SurfaceDecision

/-- Distinct latent histories can contribute nonnegative weight to one optical
    surface. Natural weights represent finite counts or common-denominator masses. -/
def hit [DecidableEq A] (s : A) : List (A × Nat) → Nat
  | [] => 0
  | (v,n) :: rest => (if v = s then n else 0) + hit s rest

def miss [DecidableEq A] (s : A) : List (A × Nat) → Nat
  | [] => 0
  | (v,n) :: rest => (if v = s then 0 else n) + miss s rest

def total : List (A × Nat) → Nat
  | [] => 0
  | (_,n) :: rest => n + total rest

theorem partition [DecidableEq A] (s : A) (d : List (A × Nat)) :
    hit s d + miss s d = total d := by
  induction d with
  | nil => rfl
  | cons a d ih =>
    rcases a with ⟨v,n⟩
    by_cases h : v = s <;> simp [hit, miss, total, h] <;> omega

/-- Exact one-span loss difference; choosing after pooling competing lexical
    histories is Bayes-optimal for surface error under these masses. -/
theorem surface_regret [DecidableEq A] (d : List (A × Nat)) (best alternative : A)
    (wins : hit alternative d ≤ hit best d) :
    miss alternative d = miss best d + (hit best d - hit alternative d) := by
  have h := partition best d
  have k := partition alternative d
  omega

/-- An explicitly testable numerical margin for preserving a decision when
    replacing contextual probability masses by estimated surface masses. -/
theorem stable [DecidableEq A] (d e : List (A × Nat)) (best other : A) (tol : Nat)
    (m : hit other d + 2 * tol < hit best d)
    (low : hit best d ≤ hit best e + tol) (high : hit other e ≤ hit other d + tol) :
    hit other e < hit best e := by omega

#print axioms surface_regret
#print axioms stable
end SurfaceDecision
