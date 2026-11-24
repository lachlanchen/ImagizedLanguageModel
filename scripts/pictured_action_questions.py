"""A full English query is rendered as ink; the source and the reply carry no IDs."""

TRAIN_FRAMES=[
    "What did they {}?", "Whom or what did they {}?", "What do they {} in the passage?",
    "Which thing do they {}?", "When they {}, what do they act on?",
    "Give the object of the action '{}'.", "For '{}', name what receives the action.",
    "What is the target when they {}?", "When someone happens to {}, name the object.",
    "If the action is '{}', give its object in English.",
]
HELD_FRAMES=["What is it that they {} here?", "In the line about the action '{}', what is affected?"]
