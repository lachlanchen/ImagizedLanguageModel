import Std

namespace PairedLibrarySampling

/-- The same two incident exposures are counted before and after a cycle
    move. The identity holds for either endpoint language and for a stop. -/
theorem balanced_vertex (a b step : Int) :
    (a + step) + (b - step) = a + b := by omega

/-- Linear scoring along a fixed update direction. Integers allow exact
    fixed-point conditional scores, gradients and indicator histograms. -/
def score (weights signal : List Int) : Int :=
  match weights, signal with
  | w :: ws, s :: ss => w*s + score ws ss
  | _, _ => 0

theorem scored_path (weights direction signal : List Int)
    (hw : weights.length = direction.length) (step : Int) :
    score (List.zipWith (fun w d => w + step*d) weights direction) signal =
      score weights signal + step * score direction signal := by
  induction weights generalizing direction signal with
  | nil =>
    cases direction <;> simp_all [score]
  | cons w ws ih =>
    cases direction with
    | nil => simp at hw
    | cons d ds =>
      have rest : ws.length = ds.length := by simpa using hw
      cases signal with
      | nil => simp [score, List.zipWith]
      | cons s ss =>
        simp only [List.zipWith, score]
        rw [ih ds ss rest]
        simp only [Int.add_mul, Int.mul_add, Int.mul_assoc]
        omega

/-- With probabilities down/(up+down) and up/(up+down), any fixed linear
    score has its old conditional expectation. Denominators are cleared,
    so this applies directly to the integer sampler and every example. -/
theorem fair_choice (old slope up down : Int) :
    down * (old + up*slope) + up * (old - down*slope) = (down + up)*old := by
  simp only [Int.mul_add, Int.mul_sub, Int.add_mul, ← Int.mul_assoc]
  rw [Int.mul_comm down up]
  omega

theorem preserved_margin (weights direction signal : List Int)
    (hw : weights.length = direction.length) (step : Int)
    (zero : score direction signal = 0) :
    score (List.zipWith (fun w d => w + step*d) weights direction) signal =
    score weights signal := by
  rw [scored_path weights direction signal hw step, zero]
  simp

end PairedLibrarySampling
