"""Evaluation-only wordings reserved before generating the new training questions."""

# This set was written for this replication. The published classroom questions
# and the older equal-trigram order controls are scored separately.
HELD_PAIRS = [
    ("What act does this line attribute to someone?", "Which person or thing is on the receiving end of that act?"),
    ("Give the deed mentioned by the narrator.", "Give the recipient of the deed mentioned by the narrator."),
    ("In plain English, what did somebody do in the passage?", "In plain English, whom or what did somebody act upon in the passage?"),
    ("Tell me how the person acted.", "Tell me whom or what the person acted on."),
    ("Which word tells us what was done?", "Which word tells us what it was done to?"),
    ("I want the action, leaving its target aside.", "I want its target, leaving the action aside."),
    ("這段記載說的是何種舉動？", "這段記載中，這個舉動是對誰或何物做的？"),
    ("請指出當事人所做的事。", "請指出承受此事的人或物。"),
    ("句中人究竟幹了甚麼？", "句中人究竟對誰或甚麼東西下手？"),
    ("我想知道發生了哪種行動。", "我想知道哪個對象受到那個行動的影響。"),
    ("用英文講講這人怎麼做。", "用英文講講這人是對什麼做的。"),
    ("请写出所做的事，不必写对象。", "请写出针对的对象，不必写所做的事。"),
]
