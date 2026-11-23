import Std
namespace ParagraphEvidence

structure Trial where
  selectedRight : Bool
  rightReply : Bool
  rivalReply : Bool

def routed (x : Trial) : Bool := if x.selectedRight then x.rightReply else x.rivalReply

def errors (pass : Trial → Bool) : List Trial → Nat
  | [] => 0
  | x :: xs => (if pass x then 0 else 1) + errors pass xs

/-- Choosing the annotated paragraph and answering it correctly suffice for a
correct reply, even when a rival paragraph can coincidentally give the answer. -/
theorem error_budget (xs : List Trial) :
    errors routed xs ≤ errors Trial.selectedRight xs + errors Trial.rightReply xs := by
  induction xs with
  | nil => decide
  | cons x xs ih =>
    obtain ⟨a,b,c⟩ := x
    cases a <;> cases b <;> cases c <;> simp [errors,routed] at * <;> omega

/-- A better selector cannot fix a query that both possible writer calls miss. -/
theorem neither_source (x : Trial) (a : x.rightReply = false) (b : x.rivalReply = false) :
    routed x = false := by simp [routed,a,b]

#print axioms error_budget
#print axioms neither_source
end ParagraphEvidence
