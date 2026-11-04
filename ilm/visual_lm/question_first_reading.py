"""Host-side prompt ordering; the model receives the resulting images only."""

from __future__ import annotations


def ordered_question(record, order):
    if order not in ("context", "question"):
        raise ValueError("unknown prompt order")
    context, question = record["context"], record["question"]
    expected = f"资料：{context}问题：{question}回答："
    if record["prompt"] != expected:
        raise ValueError("unexpected prepared prompt layout")
    prompt = expected if order == "context" else f"问题：{question}资料：{context}回答："
    if len(prompt) != len(expected):
        raise AssertionError("prompt order changed visible length")
    return {**record, "prompt": prompt}


def blank_question_images(images, record, order):
    """Remove actual question pixels without altering its length or passage."""
    ordered = ordered_question(record, order)
    start = 6 + len(record["context"]) if order == "context" else 3
    if ordered["prompt"][start:start + len(record["question"])] != record["question"]:
        raise AssertionError("question range mismatch")
    if len(images) != len(ordered["prompt"]):
        raise ValueError("prompt image length differs")
    images = images.clone()
    images[start:start + len(record["question"])] = 0
    return images
