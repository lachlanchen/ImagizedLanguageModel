"""Publish family identities before fitting the two endpoint corrections."""

from scripts.prepare_competing_facts import ROOT, reserve, read, dump, sha

WORK = ROOT / "artifacts/endpoint_interaction_v1"
PUB = ROOT / "references/evidence/endpoint_interaction_v1"
MODEL = ROOT / "models/endpoint_interaction"


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    PUB.mkdir(parents=True, exist_ok=True)
    receipt = PUB / "split.json"
    held = WORK / "held.json"
    if receipt.exists():
        if not held.exists() or sha(held) != read(receipt)["sha256"]:
            raise ValueError("The reserved pages changed")
        return
    pages = reserve("endpoint-interaction-v1", 32, ("competing_facts_v1/held.json",))
    dump(held, pages)
    dump(
        receipt,
        dict(
            sha256=sha(held),
            docs=[r["doc"] for r in pages],
            protocol=sha(ROOT / "references/endpoint_interaction_protocol.md"),
        ),
    )


if __name__ == "__main__":
    main()
