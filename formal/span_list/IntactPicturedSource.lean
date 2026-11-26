import Std
namespace IntactPicturedSource

/-- A learned marker depends only on the input pixels. -/
def augment (mark : X → M) (x : X) := (x, mark x)

/-- Soft marks may help a restricted learner but cannot create new observations.
In particular, adding them cannot collapse a distinction between full inputs. -/
theorem equality_iff (mark : X → M) (p q : X) :
    augment mark p = augment mark q ↔ p = q := by
  constructor
  · exact fun h => congrArg Prod.fst h
  · exact fun h => congrArg (augment mark) h

def cost (writer : X → A) (loss : A → A → Nat) : List (X × A) → Nat
  | [] => 0
  | (x,y) :: rest => loss (writer x) y + cost writer loss rest

/-- Every augmented writer can be realized on the original observation at exactly
the same finite-sample loss. Conversely, every original writer has a lifted
augmented writer. No assumption on labels, loss, or learnability is necessary. -/
theorem attainable_risks (mark : X → M) (data : List (X × A))
    (loss : A → A → Nat) (k : Nat) :
    (∃ w : X × M → A, cost (fun x => w (augment mark x)) loss data = k) ↔
    (∃ w : X → A, cost w loss data = k) := by
  constructor
  · rintro ⟨w,h⟩
    exact ⟨fun x => w (augment mark x), h⟩
  · rintro ⟨w,h⟩
    exact ⟨fun z => w z.1, h⟩

end IntactPicturedSource
