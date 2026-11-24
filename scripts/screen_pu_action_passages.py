"""Fix a genuinely unseen literary page set before running any reader on it."""
from functools import lru_cache
from scripts.prepare_visual_action_meaning import ROOT,WORK,RAW,read,dump,sha,table
from scripts.prepare_pronoun_writer import window
from scripts.train_typeface_writer import FACES,Face


def screen():
    face=Face(FACES[0]);alphabet={}
    @lru_cache(None)
    def cell(c):return alphabet.setdefault(face.cell(c),len(alphabet))
    def key(s):return tuple(cell(c) for c in s)
    src=RAW/'pu.txt';text=src.read_text();eng=RAW/'giles.txt';en=eng.read_text()
    contents=table('pu.tsv')
    old=read(ROOT/'artifacts/visual_word_senses_v1/train.json')
    windows={key(l.rjust(8)+c+r.ljust(8)) for _,_,_,l,c,r,_ in old}
    windows.update(key(row[0]) for row in read(ROOT/'artifacts/pronoun_writer_v1/train.json'))
    syntax=read(ROOT/'artifacts/visual_clause_roles_v1/train.json')
    pages={key(r['page']) for r in syntax}
    windows.update(key(window(r['page'],j)) for r in syntax for j in r['focus'])
    stories={'勞山道士':'THE TAOIST PRIEST OF LAO-SHAN.', '種梨':'PLANTING A PEAR-TREE.', '長清僧':'THE BUDDHIST PRIEST OF CH‘ANG-CH‘ING.', '畫皮':'THE PAINTED SKIN.', '偷桃':'THEFT OF THE PEACH.'}
    rows=[]
    for r in contents:
        page=r['line'];story=r['story']
        source=text.split('〈'+story+'〉',1)[1].split('〈',1)[0]
        # Bare Chinese text with punctuation elided, preserving the original order.
        assert page in ''.join(c for c in source if '\u3400'<=c<='\u9fff') and stories[story] in en,(story,page)
        focus=[page.index(r[k]) for k in ('verb','object')]
        flags=[key(window(page,j)) in windows for j in focus]
        rows.append(dict(**r,focus=focus,page_fit=key(page) in pages,window_fit=flags,clean=not(any(flags) or key(page) in pages)))
    pairs=list(zip(rows[::2],rows[1::2]));accepted=[i for i,(a,b) in enumerate(pairs) if a['clean'] and b['clean']]
    dump(WORK/'pu_screen.json',dict(rows=rows,clean_pair_indices=accepted,chinese=sha(src),translation=sha(eng)))
    print(len(accepted),'clean of',len(pairs))

if __name__=='__main__':screen()
