import Std
namespace RecurrentEvidencePaths
variable {G : Type} [DecidableEq G]

/-- Return the still-unobserved suffix after printing an entire source run.
Once the observation is exhausted, subsequent output remains unobserved. -/
def consume : List G → List G → Option (List G)
  | [], _ => some []
  | g :: y, [] => some (g :: y)
  | g :: y, h :: s => if g = h then consume y s else none

theorem consume_append (obs x z : List G) :
    consume obs (x ++ z) = (consume obs x).bind (fun r => consume r z) := by
  induction x generalizing obs with
  | nil => cases obs <;> simp [consume]
  | cons a x ih =>
    cases obs with
    | nil => simp [consume]
    | cons b y => by_cases h : b=a <;> simp [consume,h,ih]

inductive Tree (G : Type)
  | stop (weight : Nat)
  | choice (chunk : List G) (weight : Nat) (suffix alternatives : Tree G)

def paths : Tree G → List (List G × Nat)
  | .stop w => [([],w)]
  | .choice x w rest alt =>
    (paths rest).map (fun (s,p) => (x ++ s,w*p)) ++ paths alt

def count (obs : List G) : List (List G × Nat) → Nat
  | [] => 0
  | (s,p)::xs => (if consume obs s = some [] then p else 0) + count obs xs

theorem count_append (obs : List G) (xs zs : List (List G × Nat)) :
    count obs (xs ++ zs) = count obs xs + count obs zs := by
  induction xs with
  | nil => simp [count]
  | cons x xs ih => obtain ⟨s,p⟩ := x; simp [count, ih, Nat.add_assoc]

theorem prepend_count (obs x : List G) (w : Nat) (xs : List (List G × Nat)) :
    count obs (xs.map (fun (s,p) => (x++s,w*p))) =
      match consume obs x with
      | none => 0
      | some r => w * count r xs := by
  induction xs with
  | nil => cases h : consume obs x <;> simp [count,h]
  | cons v xs ih =>
    obtain ⟨s,p⟩ := v
    cases h : consume obs x with
    | none => simp [count,consume_append,h,ih]
    | some r =>
      by_cases c : consume r s = some [] <;>
        simp [count,consume_append,h,c,ih,Nat.mul_add]

/-- Inference over all run boundaries. The stop event can itself be included
as a final observed mark, or the observation can be an unfinished prefix. -/
def inside (obs : List G) : Tree G → Nat
  | .stop w => if obs.isEmpty then w else 0
  | .choice x w rest alt =>
    (match consume obs x with
    | none => 0
    | some r => w * inside r rest) + inside obs alt

theorem inside_is_observed_path_mass (t : Tree G) (obs : List G) :
    inside obs t = count obs (paths t) := by
  induction t generalizing obs with
  | stop w => cases obs <;> simp [inside,paths,count,consume]
  | choice x w rest alt ir ia =>
    simp only [inside,paths,count_append,prepend_count]
    cases consume obs x <;> simp [ir,ia]

#print axioms inside_is_observed_path_mass
end RecurrentEvidencePaths
