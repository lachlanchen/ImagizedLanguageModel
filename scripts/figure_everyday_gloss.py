"""Build the figure's editable data from the archived model outputs."""

from scripts.evaluate_everyday_gloss import ARMS, DEST
from scripts.prepare_asked_gloss import dump, read

def main():
    result = read(DEST / "result.json")["totals"]
    broad = result["everyday"]
    rows = [["Books and word order", "Passages", "Located", "Located + written", "Writer ceiling"]]
    for key, label in [("previous", "Earlier held books"), ("fresh", "Two new books"),
                       ("fresh/fronted", "Object before verb"), ("fresh/postposed", "Verb before object")]:
        s = broad[key]
        rows.append([label] + [str(s[k]) for k in ("pages","classroom_focus","classroom","writable")])
    dump(DEST / "figure.json", dict(table=rows,
         series=[[result[arm]["previous"]["pairs"][0],result[arm]["previous"]["pairs"][1],
                  result[arm]["previous"]["classroom"]] for arm in ARMS]))

if __name__ == "__main__":
    main()
