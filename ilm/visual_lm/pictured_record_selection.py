"""Locate an explicitly named headword on a card of visually delimited records."""
import numpy as np


def select_visible_record(passage, question, opening, closing):
    """Use a unique pictured 【head】 match or return the complete source unchanged.

    Rows may be local integer bitmap addresses or byte arrays of packed pixels.
    All comparisons require exact physical equality.
    """
    source = np.ascontiguousarray(passage)
    query = np.ascontiguousarray(question)
    if not len(source) or not len(query):
        return source, "empty"
    row_type = np.dtype((np.void, source[0].nbytes))
    p, q = source.view(row_type).reshape(-1), query.view(row_type).reshape(-1)
    left = np.asarray(opening, dtype=source.dtype).tobytes()
    right = np.asarray(closing, dtype=source.dtype).tobytes()
    headers = [i for i in range(len(p)-2) if p[i].tobytes() == left and p[i+2].tobytes() == right]
    if not headers or headers[0] != 0:
        return source, "unstructured"
    visible = {x.tobytes() for x in q}
    found = [k for k, i in enumerate(headers) if p[i+1].tobytes() in visible]
    if len(found) != 1:
        return source, "ambiguous" if found else "unmatched"
    k = found[0]
    end = headers[k+1] if k+1 < len(headers) else len(p)
    return source[headers[k]:end], "located"
