import Std

/-! Observable predictions for a restricted language of pictured factual statements.
A glyph may be any value, including a packed bitmap without a character code. -/
namespace PicturedFacts

structure Fact (G R : Type) where
  subject : List G
  relation : R
  object : List G
  deriving DecidableEq

def render (frames : R → List G × List G × List G) (f : Fact G R) : List G :=
  let (opening, link, ending) := frames f.relation
  opening ++ f.subject ++ link ++ f.object ++ ending

theorem correct_statement (frames : R → List G × List G × List G)
    (candidate target : Fact G R) (s : candidate.subject = target.subject)
    (r : candidate.relation = target.relation) (o : candidate.object = target.object) :
    render frames candidate = render frames target := by
  cases candidate
  cases target
  simp_all [render]

/-- If a representation erases a distinction, no deterministic decoder can answer both. -/
theorem collision (encode : X → Z) (decode : Z → Y) (target : X → Y) (x x' : X)
    (same : encode x = encode x') (different : target x ≠ target x') :
    ¬ (decode (encode x) = target x ∧ decode (encode x') = target x') := by
  rintro ⟨h, h'⟩
  exact different (h.symm.trans ((congrArg decode same).trans h'))

/-- Agreement on covered representatives extends to their transforms when both laws
    of equivariance hold. Neither the coverage nor the learned law is presumed in a test. -/
theorem coverage (predict truth : X → Y) (move : A → X → X) (act : A → Y → Y)
    (seen : X → Prop) (fit : ∀ x, seen x → predict x = truth x)
    (modelLaw : ∀ a x, predict (move a x) = act a (predict x))
    (truthLaw : ∀ a x, truth (move a x) = act a (truth x))
    (x : X) (covered : ∃ a z, seen z ∧ x = move a z) : predict x = truth x := by
  obtain ⟨a,z,h,rfl⟩ := covered
  rw [modelLaw,truthLaw,fit z h]

def miss [DecidableEq A] (a b : A) : Nat := if a = b then 0 else 1

def total (score : X → Nat) : List X → Nat
  | [] => 0
  | x :: xs => score x + total score xs

/-- Complete answer errors on an arbitrary finite corpus are bounded by the sum of
    subject, relation, and object errors. Multiple errors can occur on one example. -/
theorem corpus_bound [DecidableEq G] [DecidableEq R]
    (frames : R → List G × List G × List G) (pairs : List (Fact G R × Fact G R)) :
    total (fun (a,b) => miss (render frames a) (render frames b)) pairs ≤
    total (fun (a,b) => miss a.subject b.subject + miss a.relation b.relation +
      miss a.object b.object) pairs := by
  induction pairs with
  | nil => simp [total]
  | cons pair rest ih =>
    obtain ⟨a,b⟩ := pair
    have one : miss (render frames a) (render frames b) ≤
        miss a.subject b.subject + miss a.relation b.relation + miss a.object b.object := by
      by_cases s : a.subject = b.subject <;> by_cases r : a.relation = b.relation <;>
        by_cases o : a.object = b.object <;>
        simp [miss,s,r,o,correct_statement frames a b] <;> split <;> omega
    simpa only [total] using Nat.add_le_add one ih

/-- Learn a spelling pattern from labelled examples. Each entry records the physical
    ink before, between, and after the two annotated participant spans. -/
def learnFrame [DecidableEq R] (r : R) : List (R × F) → Option F
  | [] => none
  | (k,f) :: xs => if k = r then some f else learnFrame r xs

theorem frame_identified [DecidableEq R] (truth : R → F) (r : R) (examples : List (R × F))
    (consistent : ∀ z ∈ examples, z.2 = truth z.1)
    (covered : ∃ f, (r,f) ∈ examples) : learnFrame r examples = some (truth r) := by
  induction examples with
  | nil => simp at covered
  | cons head tail ih =>
    obtain ⟨k,v⟩ := head
    by_cases h : k = r
    · subst k
      have t := consistent (r,v) (by simp)
      simpa [learnFrame] using congrArg some t
    · simp only [learnFrame, if_neg h]
      apply ih
      · intro z hz
        exact consistent z (by simp [hz])
      · obtain ⟨f,hf⟩ := covered
        exact ⟨f, (List.mem_cons.mp hf).resolve_left (by intro he; exact h (congrArg Prod.fst he).symm)⟩

#print axioms frame_identified
#print axioms correct_statement
#print axioms collision
#print axioms coverage
#print axioms corpus_bound
end PicturedFacts
