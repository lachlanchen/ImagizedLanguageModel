"""Derive the editable chart, table and original image strips for Figure 87."""

from PIL import Image

from scripts.audit_traditional_passage_training import archived
from scripts.infer_pictured_entry import page
from scripts.pictured_participant_scenes import fresh_cases
from scripts.prepare_pictured_grammatical_roles import RAW, records
from scripts.train_pictured_participants import PUBLIC, WORK, Tiles, load, memory_bank, treebank, write_json


def prepare():
    result = load(PUBLIC/"result.json")["arms"]
    external = load(PUBLIC/"external_referents.json")["arms"]
    bank = memory_bank()
    tile = Tiles(bank)
    pack = lambda seq: b"".join(bank.tiles[i] for i in seq)
    def outputs(tag):
        return archived(WORK/"matched"/(tag+".npz"), result["matched"][tag]["archive"])
    pairs, meta = treebank(bank,"test")
    imgs, done = outputs("gsd_test")
    for i,(p,qa,qb,a,b) in enumerate(pairs):
        if all(done[2*i:2*i+2]) and imgs[2*i:2*i+2] == [pack(p[slice(*a)]), pack(p[slice(*b)])]:
            break
    else:
        raise ValueError("No exact GSD pair")
    passages = {sid:text for sid,text,_,_ in records(RAW/"zh_gsd-ud-test.conllu")}
    source = passages[meta[i]["id"]]
    assert pack(tile(source)) == pack(p)
    scenes = list(fresh_cases())
    free_imgs, free_done = outputs("fresh")
    orig = dict(grammar=source, predicate=meta[i]["verb"],
                subject=source[slice(*a)], object=source[slice(*b)], test_row=i,
                fresh_scene=scenes[0][0], fresh_targets=[s[2] for s in scenes[:2]])
    sizes = {}
    for prefix, row in (("grammar",(pack(p),pack(qa),pack(qb),imgs[2*i],imgs[2*i+1])),
                        ("fresh",(pack(tile(scenes[0][0])),pack(tile(scenes[0][1])),
                                  pack(tile(scenes[1][1])),free_imgs[0],free_imgs[1]))):
        for name, item in zip(("source","q0","q1","a0","a1"),row,strict=True):
            im = page(item,width=34)
            key=prefix+"_"+name
            im.save(PUBLIC/(key+".png"))
            sizes[key]=list(im.size)
    table=[["Reader","New scenes / 64","New verbs / 64","Person QA / 387","XWinograd / 504"]]
    series=[]
    for label,key in (("Previous","parent"),("Exchanged","exchanged"),("Matched","matched")):
        arm=result[key]
        series.append(arm["gsd_test"]["both"])
        table.append([label,*(str(arm[k]["exact"]) for k in ("fresh","held_verbs","overlap_who")),
                      str(external[key]["exact"])])
    write_json(PUBLIC/"figure.json",dict(table=table,series=[series],dimensions=sizes,examples=orig))
    print(orig)


if __name__=="__main__":
    prepare()
