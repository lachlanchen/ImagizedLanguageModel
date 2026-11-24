"""Bind the pictured values of the two referential fields to output span actions."""
import numpy as np

ROLES = ("name", "near")


def bind_card(packed, prefixes, anchors):
    """Return normalized input ink and non-padded original cells of each bound role."""
    normalized = packed.copy()
    bound = {}
    for row_index, row in enumerate(packed):
        for role, prefix in prefixes:
            if not np.array_equal(row[:len(prefix)], prefix):
                continue
            if role in bound:
                raise ValueError("More than one value for the field "+role)
            rest = row[len(prefix):]
            used = np.flatnonzero(rest.any(axis=1))
            if not len(used):
                raise ValueError("Empty pictured "+role)
            bound[role] = rest[:used[-1]+1].copy()
            anchor = anchors[role]
            if len(prefix)+len(anchor)>len(row):
                raise ValueError("Anchor exceeds row width")
            normalized[row_index,len(prefix):] = 0
            normalized[row_index,len(prefix):len(prefix)+len(anchor)] = anchor
            break
    return normalized, bound


def expand(actions, physical, bound):
    """Return original image cells, with an error for references to absent fields."""
    output = []
    for action in actions:
        if 0 < action < len(physical):
            output.append(physical[action:action+1])
        elif len(physical) <= action < len(physical)+len(ROLES):
            role = ROLES[action-len(physical)]
            if role not in bound:
                raise ValueError("Generated an unavailable field: "+role)
            output.append(bound[role])
        else:
            raise ValueError("Invalid emitted action")
    return np.concatenate(output) if output else np.empty((0,128),np.uint8)
