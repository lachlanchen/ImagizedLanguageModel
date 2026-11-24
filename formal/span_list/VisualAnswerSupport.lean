import Std

namespace VisualAnswerSupport

/-- A single replacement can alter any interval, including an empty interval.
Left and right are the reference images around a designated occurrence. -/
theorem repair_iff {G : Type} (written left right : List G) :
    (∃ middle, written = left ++ middle ++ right) ↔
    left.length + right.length ≤ written.length ∧
    written.take left.length = left ∧
    written.drop (written.length - right.length) = right := by
  constructor
  · rintro ⟨middle, rfl⟩
    constructor
    · simp
    constructor
    · rw [List.append_assoc, List.take_left]
    · simp only [List.length_append, Nat.add_sub_cancel]
      simpa only [List.length_append] using List.drop_left (left ++ middle) right
  · rintro ⟨enough, first, last⟩
    let tail := written.drop left.length
    let cut := written.length - right.length - left.length
    refine ⟨tail.take cut, ?_⟩
    have index : cut + left.length = written.length - right.length := by
      dsimp [cut]; omega
    have suffix : tail.drop cut = right := by
      simpa only [tail, List.drop_drop, index] using last
    calc
      written = written.take left.length ++ tail := (List.take_append_drop ..).symm
      _ = left ++ tail := by rw [first]
      _ = left ++ (tail.take cut ++ tail.drop cut) := by rw [List.take_append_drop]
      _ = left ++ tail.take cut ++ right := by rw [suffix, List.append_assoc]

/-- Every actual writer step spends one decision on a drawn cell or on STOP.
Its softmax can spell from the palette and copy from a pictured source.
This is an upper bound even if the entire page is exposed to copying. -/
inductive Stopped {G : Type} (allowed : G → Prop) : Nat → List G → Prop where
  | stop {remaining} : Stopped allowed (remaining + 1) []
  | draw {remaining g gs} : allowed g → Stopped allowed remaining gs →
      Stopped allowed (remaining + 1) (g :: gs)

theorem stopped_support {G : Type} {allowed : G → Prop} {limit : Nat} {y : List G}
    (h : Stopped allowed limit y) : y.length < limit ∧ ∀ g ∈ y, allowed g := by
  induction h with
  | stop => simp
  | draw valid _ ih =>
    constructor
    · simpa only [List.length_cons, Nat.succ_lt_succ_iff] using ih.1
    · intro g hg
      rcases List.mem_cons.mp hg with rfl | tail
      · exact valid
      · exact ih.2 g tail

end VisualAnswerSupport
