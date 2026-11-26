"""Attribute every breadth-training row to the exact bytes in the upstream arrays."""
from collections import Counter
import numpy as np
from scripts.qualify_paired_translation_teacher import DATA, ROOT, PREVIOUS, PUB, base, population


def main():
    sel, _ = population()
    book = base.BOOK
    book_receipt = base.read(ROOT/'references/evidence/book_breadth_v1/corpus.json')
    optical_receipt = base.read(ROOT/'references/evidence/independent_optical_actions_v1/preparation.json')
    rehearsal = base.read(ROOT/'references/evidence/book_rehearsal_v1/design.json')
    files = {book/'cells.npy':book_receipt['files']['cells.npy'],
             book/'domains.npy':book_receipt['files']['domains.npy'],
             base.REHEARSAL:rehearsal['replacement'],base.OPTICAL:optical_receipt['files']['cells.npy']}
    assert all(base.sha(p)==h for p,h in files.items())
    by_pixels = {}
    books = np.load(book/'cells.npy',mmap_mode='r')
    domains = np.load(book/'domains.npy')
    with np.load(base.REHEARSAL) as replacements:
        replacement = dict(zip(replacements['indices'].tolist(), replacements['cells'], strict=True))
    for i,pair in enumerate(books):
        if i in replacement:
            by_pixels.setdefault(replacement[i].tobytes(),'constructed')
        else:
            by_pixels.setdefault(pair.tobytes(),('book','talk','flores')[domains[i]])
    for i,pair in enumerate(np.load(base.OPTICAL,mmap_mode='r')):
        by_pixels.setdefault(pair.tobytes(),'opus' if i < optical_receipt['opus'] else 'library')
    training = np.load(DATA/'pairs.npy',mmap_mode='r')[:sel['rows']]
    origins = [by_pixels[row.tobytes()] for row in training]
    natural_pixels = {x.tobytes() for x in books} | {x.tobytes() for x in np.load(base.OPTICAL,mmap_mode='r')}
    also_natural = sum(row.tobytes() in natural_pixels for row,src in zip(training,origins,strict=True) if src=='constructed')
    teacher = base.read(PUB/'teacher.json')
    groups = [[origins[i] for i in r['rows']] for r in sel['contrasts']]
    sets = dict(mined=range(len(groups)),audit=teacher['audit'],remaining=teacher['remaining'])
    breakdown = {k:dict(Counter(' + '.join(sorted(groups[i])) for i in ids)) for k,ids in sets.items()}
    natural = {k:[i for i in ids if 'constructed' not in groups[i]] for k,ids in sets.items()}
    N=base.read(PREVIOUS/'selection.json')['per_cohort']
    base.dump(PUB/'provenance.json',dict(files={str(p.relative_to(ROOT)):h for p,h in files.items()},
              selection=base.sha(PUB/'selection.json'),teacher=base.sha(PUB/'teacher.json'),
              broad=Counter(origins),narrow=Counter(origins[:N]),also_natural=also_natural,
              groups=groups,breakdown=breakdown,natural=natural))
    print('broad',Counter(origins),'narrow',Counter(origins[:N]),'also natural',also_natural,
          'corpus-derived families',{k:len(v) for k,v in natural.items()},flush=True)


if __name__ == '__main__':
    main()
