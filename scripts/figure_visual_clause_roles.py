"""Export the published chart and table from measured held-source responses."""
from scripts.prepare_visual_clause_roles import PUBLIC, read, dump


def main():
    data=read(PUBLIC/"transfer_result.json")
    ix=[data["names"].index(n) for n in ("previous","distilled","lexical")]
    poly=data["polysemy"]
    chart=[[poly["classroom_located"][j],poly["classroom_written"][j]] for j in ix]
    table=[["Evaluation", "Prior reader", "Distilled reader", "Glyph counts¹"]]
    groups=data["groups"]
    for label,values in [
        ("Opposite word orders (18 pairs)",data["opposite_order_written"]["classroom"]),
        ("Earlier held passages (539)",groups["prior539"]["classroom_written"]),
        ("Two earlier books (29)",groups["prior29"]["classroom_written"]),
        ("Object-first subset (14)",groups["prior29/front"]["classroom_written"])]:
        table.append([label]+[str(values[j]) for j in ix])
    dump(PUBLIC/"figure.json",dict(series=chart,table=table))


if __name__=="__main__":main()
