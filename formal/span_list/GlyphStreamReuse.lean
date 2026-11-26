import Std

namespace GlyphStreamReuse

/-- Each step may stop or choose an image code. The next decision has access
to exactly the previous state and the feedback produced from the emitted ink. -/
def run {S C P X : Type} (choose : S → Option C) (draw : C → P)
    (feedback : C → X) (step : S → X → S) : Nat → S → List P
  | 0, _ => []
  | n+1, s => match choose s with
    | none => []
    | some c => draw c :: run choose draw feedback step n (step s (feedback c))

theorem compilation_preserves_stream {S C P X : Type} (choose : S → Option C)
    (draw : C → P) (readInk : P → X) (step : S → X → S)
    (inkTable : C → P) (feedbackTable : C → X)
    (hi : ∀ c, inkTable c = draw c) (hf : ∀ c, feedbackTable c = readInk (draw c))
    (budget : Nat) (s : S) :
    run choose inkTable feedbackTable step budget s =
    run choose draw (fun c => readInk (draw c)) step budget s := by
  induction budget generalizing s with
  | zero => rfl
  | succ n ih =>
    simp only [run]
    split
    · rfl
    · simp only [hi, hf, ih]

#print axioms compilation_preserves_stream
end GlyphStreamReuse
