import Std

namespace OrderedTranslation

/-- Absent cells provide padding distinct from a pictured whitespace glyph. -/
abbrev Page (α : Type) := Nat → Option α
abbrev Blocks (α : Type) := Nat → Fin 4 → Option α

def pack (x : Page α) : Blocks α := fun b j => x (4*b+j.val)
def unpack (x : Blocks α) : Page α :=
  fun i => x (i/4) ⟨i%4,Nat.mod_lt _ (by decide)⟩

theorem recover (x : Page α) : unpack (pack x) = x := by
  funext i
  simp only [pack, unpack]
  congr 1
  omega

/-- Every rule on the uncompressed source has a rule on the packed page
with identical action, hence identical pointwise risk under any loss. -/
theorem preserve_decisions (f : Page α → β) (loss : β → γ → δ) :
    ∃ g : Blocks α → β, ∀ x target, loss (g (pack x)) target = loss (f x) target := by
  exact ⟨fun z => f (unpack z), fun x y => by simp only [recover]⟩

/-- A learned bottleneck can lose this property. A collision between pages
requiring different answers gives an error for every deterministic decoder. -/
theorem collision (enc : Blocks α → β) (decode : β → γ) (gold : Page α → γ)
    (x y : Page α) (h : enc (pack x) = enc (pack y)) (d : gold x ≠ gold y) :
    ¬ (decode (enc (pack x)) = gold x ∧ decode (enc (pack y)) = gold y) := by
  rintro ⟨hx,hy⟩
  exact d (hx.symm.trans ((congrArg decode h).trans hy))

/-- Each encoder head visits all ordered source pairs. Each decoder
cross-attention head visits each source block once per target step. -/
theorem equal_length_sites (n m : Nat) :
    (4*n)*(4*n) = 16*(n*n) ∧ m*(4*n) = 4*(m*n) := by
  constructor
  · calc (4*n)*(4*n) = (4*4)*(n*n) := by ac_rfl
         _ = 16*(n*n) := by rfl
  · ac_rfl

/-- The actual ragged layouts use ceil(n/4) valid windows. -/
theorem source_coverage (n : Nat) :
    n ≤ 4*((n+3)/4) ∧ 4*((n+3)/4) < n+4 := by omega

end OrderedTranslation

namespace OrderedTranslation

/-- The copy instruction addresses a visible specimen; its cost in head
parameters does not depend on which physical picture occupies the address. -/
inductive Action (bank page : List α) where
  | spell (i : Fin bank.length)
  | copy (i : Fin page.length)

def emit (bank page : List α) : Action bank page → α
  | .spell i => bank[i]
  | .copy i => page[i]

theorem exact_support (bank page : List α) (x : α) :
    (∃ a, emit bank page a = x) ↔ x ∈ bank ∨ x ∈ page := by
  constructor
  · rintro ⟨a, rfl⟩
    cases a with
    | spell i => exact Or.inl (List.getElem_mem bank i.val i.isLt)
    | copy i => exact Or.inr (List.getElem_mem page i.val i.isLt)
  · rintro (h | h)
    · obtain ⟨i, hi, eq⟩ := List.getElem_of_mem h
      exact ⟨.spell ⟨i,hi⟩, eq⟩
    · obtain ⟨i, hi, eq⟩ := List.getElem_of_mem h
      exact ⟨.copy ⟨i,hi⟩, eq⟩

theorem strictly_new (bank page : List α) (x : α) (visible : x ∈ page) (novel : x ∉ bank) :
    (∃ a, emit bank page a = x) ∧ ¬ (∃ i : Fin bank.length, bank[i] = x) := by
  refine ⟨(exact_support bank page x).2 (Or.inr visible), ?_⟩
  rintro ⟨i, rfl⟩
  exact novel (List.getElem_mem bank i.val i.isLt)

end OrderedTranslation

namespace OrderedTranslation

/-- An action is a fixed pictured glyph or a slot carrying an arbitrary image. -/
abbrev Plan (glyph slot : Type) := List (Sum glyph slot)

def choose (f : α → γ) (g : β → γ) : Sum α β → γ
  | .inl a => f a
  | .inr b => g b

def paint (glyph : α → γ) (payload : β → γ) (p : Plan α β) : List γ :=
  p.map (choose glyph payload)

/-- Renaming the slot alphabet cannot change the sentence surrounding the slots. -/
theorem substitute (glyph : α → γ) (payload : β → γ) (rename : δ → β)
    (p : Plan α δ) :
    paint glyph (payload ∘ rename) p =
      paint glyph payload (p.map (choose Sum.inl (Sum.inr ∘ rename))) := by
  simp only [paint, List.map_map]
  congr 1
  funext a
  cases a <;> rfl

/-- Actual bytes determine their action uniquely when fixed images and
distinct page payloads have disjoint identities. This is a per-page assumption. -/
theorem read_paint (glyph : α → γ) (payload : β → γ) (read : γ → Sum α β)
    (hg : ∀ a, read (glyph a) = Sum.inl a) (hp : ∀ b, read (payload b) = Sum.inr b)
    (p : Plan α β) : (paint glyph payload p).map read = p := by
  induction p with
  | nil => rfl
  | cons a rest ih =>
    cases a <;> simp_all [paint, choose]

/-- One exact plan comparison suffices for every injective assignment of visible
images, when the desired answer itself follows that substitution. -/
theorem same_answer (glyph : α → γ) (payload : β → γ) (read : γ → Sum α β)
    (hg : ∀ a, read (glyph a) = Sum.inl a) (hp : ∀ b, read (payload b) = Sum.inr b)
    (pred gold : Plan α β) :
    paint glyph payload pred = paint glyph payload gold ↔ pred = gold := by
  constructor
  · intro h
    have hh := congrArg (List.map read) h
    simpa only [read_paint glyph payload read hg hp] using hh
  · rintro rfl
    rfl

/-- On a mixed binary label class collapsed to one representation, any
deterministic answer incurs at least the minority mass in mistakes, and a
majority decision attains it. The counts can also be unnormalised weights. -/
def mistakes (positive negative : Nat) (prediction : Bool) : Nat :=
  if prediction then negative else positive

theorem optimal_collapsed_error (p n : Nat) :
    (∀ b, min p n ≤ mistakes p n b) ∧
    ∃ b, mistakes p n b = min p n := by
  constructor
  · intro b
    cases b
    · exact Nat.min_le_left p n
    · exact Nat.min_le_right p n
  · by_cases h : p ≤ n
    · exact ⟨false, by simp [mistakes, Nat.min_eq_left h]⟩
    · exact ⟨true, by simp [mistakes, Nat.min_eq_right (by omega : n ≤ p)]⟩

end OrderedTranslation

namespace OrderedTranslation

/-- For arbitrary noiseless binary targets, observations identify the entire
function if and only if every distinct input has been observed. -/
theorem identifies_iff_covers {X : Type} [DecidableEq X] (seen : X → Prop) :
    (∀ f g : X → Bool, (∀ x, seen x → f x = g x) → f = g) ↔ ∀ x, seen x := by
  constructor
  · intro ident x
    by_cases h : seen x
    · exact h
    · have agree : ∀ z, seen z → false = decide (z = x) := by
        intro z hz
        by_cases same : z = x
        · subst z
          exact False.elim (h hz)
        · simp [same]
      have eq := congrFun (ident (fun _ => false) (fun z => decide (z = x)) agree) x
      simp at eq
  · intro cover f g agree
    exact funext (fun x => agree x (cover x))

/-- A demonstrated substitution-invariant decision needs a label for each
context; a labelled example with any payload suffices for that context. -/
theorem invariant_identification {C S : Type} [DecidableEq C] (seen : C → S → Prop) :
    (∀ f g : C → Bool, (∀ c s, seen c s → f c = g c) → f = g) ↔
      ∀ c, ∃ s, seen c s := by
  constructor
  · intro h
    apply (identifies_iff_covers (fun c => ∃ s, seen c s)).mp
    intro f g agree
    exact h f g (fun c s hs => agree c ⟨s, hs⟩)
  · intro cover f g agree
    funext c
    obtain ⟨s, hs⟩ := cover c
    exact agree c s hs

end OrderedTranslation
