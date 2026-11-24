import Std

namespace VisualClassTransfer

/-- A labelled prompt may contain a passage as well as a pictured question.
The target may be a complete written reply. The key can be computed from all pixels. -/
def fit {I K A : Type} [DecidableEq K] (key : I → K) :
    List (I × A) → I → Option A
  | [], _ => none
  | (x,a)::rest, q => if key x = key q then some a else fit key rest q

/-- One correctly labelled representative of each encountered class suffices
for exact transfer if its context-dependent key preserves the required answer. -/
theorem transfer {I K A : Type} [DecidableEq K] (key : I → K) (meaning : K → A)
    (examples : List (I × A)) (q : I)
    (consistent : ∀ s ∈ examples, s.2 = meaning (key s.1))
    (covered : ∃ s ∈ examples, key s.1 = key q) :
    fit key examples q = some (meaning (key q)) := by
  induction examples with
  | nil => simp at covered
  | cons s ss ih =>
    obtain ⟨x,a⟩ := s
    by_cases same : key x = key q
    · have label := consistent (x,a) (by simp)
      simpa [fit,same] using congrArg some (label.trans (congrArg meaning same))
    · simp only [fit,if_neg same]
      apply ih
      · intro z mem; exact consistent z (List.mem_cons_of_mem _ mem)
      · obtain ⟨z,hz,eq⟩ := covered
        rcases List.mem_cons.mp hz with h | mem
        · subst z; exact False.elim (same eq)
        · exact ⟨z,mem,eq⟩

/-- Permuting the unknown classes of two previously unseen opaque identities
leaves all observed labels identical. A predictor given only those same labelled
observations and the same query must miss one of the two possible answers. -/
theorem opaqueAmbiguity {T K A : Type} [DecidableEq T]
    (key : T → K) (meaning : K → A) (seen : List T) (u v : T)
    (freshU : u ∉ seen) (freshV : v ∉ seen)
    (different : meaning (key u) ≠ meaning (key v))
    (learner : List (T × A) → T → A) :
    let alt := fun t => if t = u then key v else if t = v then key u else key t
    let observations := fun (k : T → K) => seen.map (fun t => (t,meaning (k t)))
    observations key = observations alt ∧
    (learner (observations key) u ≠ meaning (key u) ∨
     learner (observations alt) u ≠ meaning (alt u)) := by
  dsimp only
  have common : seen.map (fun t => (t,meaning (key t))) =
      seen.map (fun t => (t,meaning (if t = u then key v else if t = v then key u else key t))) := by
    apply List.map_congr_left
    intro t ht
    have hu : t ≠ u := fun h => freshU (h ▸ ht)
    have hv : t ≠ v := fun h => freshV (h ▸ ht)
    simp [hu,hv]
  refine ⟨common,?_⟩
  rw [←common]
  simp only [if_pos rfl]
  by_cases h : learner (seen.map fun t => (t,meaning (key t))) u = meaning (key u)
  · right; exact fun h' => different (h.symm.trans h')
  · exact Or.inl h
end VisualClassTransfer
