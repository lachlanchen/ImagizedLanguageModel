import Std

namespace PrefixRecovery

/-- A monotone alignment of two streams of physical shapes.
    Its third index counts insertions, deletions and substitutions. -/
inductive Align {α : Type} : List α → List α → Nat → Prop where
  | empty : Align [] [] 0
  | keep (x : α) : Align a b n → Align (x :: a) (x :: b) n
  | change (x y : α) : Align a b n → Align (x :: a) (y :: b) (n+1)
  | drop (x : α) : Align a b n → Align (x :: a) b (n+1)
  | insert (y : α) : Align a b n → Align a (y :: b) (n+1)

theorem length_bound {a b : List α} (h : Align a b n) : b.length ≤ a.length+n := by
  induction h <;> simp_all <;> omega

theorem erase (a : List α) : Align a [] a.length := by
  induction a with
  | nil => exact .empty
  | cons x xs ih => exact .drop x ih

theorem exact_tail (a : List α) : Align a a 0 := by
  induction a with
  | nil => exact .empty
  | cons x xs ih => exact .keep x ih

theorem copy_suffix {a b : List α} (h : Align a b n) (tail : List α) :
    Align (a++tail) (b++tail) n := by
  induction h with
  | empty => exact exact_tail tail
  | keep x _ ih => exact .keep x ih
  | change x y _ ih => exact .change x y ih
  | drop x _ ih => exact .drop x ih
  | insert y _ ih => exact .insert y ih

/-- Cut an arbitrary alignment where its source first finishes the fixed draft. -/
theorem cut_alignment {a b : List α} (h : Align a b n) :
    ∀ p u, a = p++u → ∃ v w k, b = v++w ∧ Align p v k ∧ k ≤ n := by
  induction h with
  | empty =>
    intro p u hp
    cases p with
    | nil => exact ⟨[],[],0,rfl,.empty,Nat.le_refl _⟩
    | cons x xs => simp at hp
  | keep x h ih =>
    intro p u hp
    cases p with
    | nil => exact ⟨[],_,0,rfl,.empty,Nat.zero_le _⟩
    | cons z zs =>
      obtain ⟨rfl, hp'⟩ := List.cons.inj hp
      obtain ⟨v,w,k,rfl,hv,hk⟩ := ih zs u hp'
      exact ⟨_::v,w,k,rfl,.keep _ hv,hk⟩
  | change x y h ih =>
    intro p u hp
    cases p with
    | nil => exact ⟨[],_,0,rfl,.empty,Nat.zero_le _⟩
    | cons z zs =>
      obtain ⟨rfl,hp'⟩ := List.cons.inj hp
      obtain ⟨v,w,k,rfl,hv,hk⟩ := ih zs u hp'
      exact ⟨y::v,w,k+1,rfl,.change _ _ hv,by omega⟩
  | drop x h ih =>
    intro p u hp
    cases p with
    | nil => exact ⟨[],_,0,rfl,.empty,Nat.zero_le _⟩
    | cons z zs =>
      obtain ⟨rfl,hp'⟩ := List.cons.inj hp
      obtain ⟨v,w,k,rfl,hv,hk⟩ := ih zs u hp'
      exact ⟨v,w,k+1,rfl,.drop _ hv,by omega⟩
  | insert y h ih =>
    intro p u hp
    obtain ⟨v,w,k,rfl,hv,hk⟩ := ih p u hp
    exact ⟨y::v,w,k+1,rfl,.insert _ hv,by omega⟩

/-- Exact reachability with an edit budget after the writer has committed p.
    The suffix of a minimizing alignment supplies an optimal completion. -/
theorem completion_iff {p y : List α} {budget : Nat} :
    (∃ tail n, Align (p++tail) y n ∧ n ≤ budget) ↔
    (∃ v w k, y = v++w ∧ Align p v k ∧ k ≤ budget) := by
  constructor
  · rintro ⟨tail,n,h,hn⟩
    obtain ⟨v,w,k,hy,hv,hk⟩ := cut_alignment h p tail rfl
    exact ⟨v,w,k,hy,hv,by omega⟩
  · rintro ⟨v,w,k,rfl,hv,hk⟩
    exact ⟨w,k,copy_suffix hv w,hk⟩

/-- Erasing the draft costs its length, so any minimizing reference prefix lies
    inside the doubled length search window used by the dynamic program. -/
theorem endpoint_window {p v : List α} (h : Align p v k) (hk : k ≤ p.length) :
    v.length ≤ 2*p.length := by
  have := length_bound h
  omega

end PrefixRecovery
