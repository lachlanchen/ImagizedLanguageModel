"""Training-only bilingual renderings and independent, fixed evaluation questions."""

from scripts.asked_gloss_questions import TRAIN, HELD
from scripts.probe_asked_gloss_paraphrases import PARAPHRASES

# Matched conversational pairs. Their order encodes training targets, never inference.
BROAD = [
    ("What did they do?", "What did they do it to?"),
    ("What happened in the story?", "To whom did it happen?"),
    ("What did the person actually do?", "What person or thing did they act on?"),
    ("How did the subject act?", "Which thing did the subject affect?"),
    ("Which deed does the passage describe?", "Which person or thing is affected by that deed?"),
    ("In a word, what did someone do?", "In a word, what did someone handle or affect?"),
    ("Tell me the action.", "Tell me the target of the action."),
    ("What is the speaker doing in this line?", "Who or what is the speaker acting on?"),
    ("Give me the activity in English.", "Give me the recipient of the activity in English."),
    ("What takes place?", "What undergoes the change?"),
    ("Which action is described here?", "What receives that action here?"),
    ("Name the thing the person does.", "Name the thing the person acts upon."),
    ("他做了甚麼事", "他對誰或甚麼東西做這件事"),
    ("文中的人做了哪件事", "這件事影響了誰或哪樣東西"),
    ("說出人物的動作", "說出承受動作的人或物"),
    ("發生了哪種行動", "行動指向的是誰或什麼"),
    ("他在做什麼事呢", "他對什麼做這件事呢"),
    ("告訴我文中的行為", "告訴我這個行為影響的人或物"),
    ("這裡描寫哪種舉動", "這一舉動施加在哪個對象身上"),
    ("請指出他幹了什麼", "請指出他針對什麼人或事物"),
    ("句中人物做了哪個動作", "句中人物的動作作用於什麼"),
    ("用一個詞說他做了啥", "用一個詞說他作用於誰或什麼"),
    ("他做了什么事", "他对谁或什么东西做这件事"),
    ("这里发生了什么动作", "这个动作影响了谁或什么"),
]
PUBLISHED = [
    ("What is the subject doing?", "Who or what receives the action of the verb?"),
    ("做什麼？", "動作是針對誰或什麼？"),
]
TEST = PUBLISHED + PARAPHRASES + HELD
LABELS = ["classroom_en", "classroom_zh", "everyday_zh1", "everyday_en1",
          "everyday_zh2", "everyday_en2", "ordered_zh1", "ordered_zh2",
          "ordered_en1", "ordered_en2"]
