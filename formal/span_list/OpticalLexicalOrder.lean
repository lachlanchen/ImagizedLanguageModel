import Std
namespace OpticalLexicalOrder

/-- Unnormalised source-alignment mass for one target strip, in fixed-point units. -/
def mass (weight : S → T → Nat) (target : T) : List S → Nat
  | [] => 0
  | s :: rest => weight s target + mass weight target rest

theorem bag_invariant (weight : S → T → Nat) (t : T)
    {xs ys : List S} (h : List.Perm xs ys) : mass weight t xs = mass weight t ys := by
  induction h with
  | nil => rfl
  | cons a _ ih => simp [mass, ih]
  | swap a b rest => simp [mass, Nat.add_left_comm]
  | trans _ _ ih₁ ih₂ => exact ih₁.trans ih₂

/-- Including the null alignment and averaging over the source leaves the
same numerator and denominator for any permutation of its visual strips. -/
theorem mixture_invariant (weight : S → T → Nat) (null : T → Nat) (t : T)
    {xs ys : List S} (h : xs.Perm ys) :
    (null t + mass weight t xs, xs.length + 1) =
    (null t + mass weight t ys, ys.length + 1) := by
  rw [bag_invariant weight t h, h.length_eq]
end OpticalLexicalOrder
