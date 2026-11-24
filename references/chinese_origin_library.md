# A searchable library of Chinese historical forms

The local SQLite collection is `data/chinese_origin_library_v1/origin.sqlite3`. It indexes every retained query from the local Hanziyuan collection, preserves alternate entries under the same character, and stores the original vector image bytes. The collection contains 9,055 queried headwords. Of these, 7,755 yielded entries and 1,300 have a recorded absence. This coverage describes the local snapshot. New excavations, other catalogues and headwords outside the original query list are outside its scope. [Inventory and database fingerprint](evidence/origin_library_v1/inventory.json)

| Source catalogue section | Entry-to-form links |
|---|---:|
| Oracle bone | 22,415 |
| Bronze | 18,725 |
| Seal | 5,434 |
| Liushutong | 22,176 |
| Total | 68,750 |

The 8,594 entries link to 65,769 distinct SVGs. A catalogue section identifies the source's grouping. Dates, provenances and chronological sequences require supporting evidence for individual forms. Five bronze links have visible vector art that disappears under the 32-pixel rasterization; their raw SVGs remain available. No identical SVG hash crosses the train, development and sealed partitions. Related writing styles, common modern characters and linguistic information may still recur across partitions. [Builder](../scripts/build_origin_library.py) · [Catalogue interface](https://hanziyuan.net/)

The book index covers 11 supplied volumes and 9,755 pages, with source hashes and one-based page numbers. A PDF page renders from its original file. The reflowable EPUB uses a 612-by-792-point page at a 12-point layout size. Native text extraction can return an empty string on a scanned page. Book files stay in the private local library. The writer training described here uses the authenticated catalogue forms; the book index supports inspection and later page-grounded work. [Book catalogue](source_books/README.md)

From the repository root, select a fresh output directory for each export.

```bash
python -m scripts.read_origin_library --char 言 --output /tmp/ilm-yan-forms
python -m scripts.read_origin_library --book oracle-bone-dictionary --page 7 --output /tmp/ilm-oracle-p7
python -m scripts.build_origin_library --legacy ../incoder --books references/source_books/library
```

The first command writes an entry manifest with source fields and labelled, checksum-verified SVG files. The second writes the rendered page and its provenance. The third recreates the database from the retained private source materials and verifies compatibility with any existing collection.
