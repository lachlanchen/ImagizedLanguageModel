"""Source-order contrast during generation from the same set of pictured cells."""

import math
import torch


@torch.inference_mode()
def write_with_order(
    writer, features, source, inventory, direction, physical_rank, limit=128
):
    if physical_rank.shape != (len(features),):
        raise ValueError("Missing physical image ordering")
    pad = source.eq(0)
    order = (
        physical_rank[source]
        .masked_fill(pad, len(features))
        .argsort(dim=1, stable=True)
    )
    sorted_cells = source.gather(1, order)
    bank = writer.optical(features)
    context, valid = writer.encode(bank, torch.cat([source, sorted_cells]))
    keys = writer.prototypes(bank, inventory)
    size = len(source)
    x = writer.begin[direction].expand(2 * size, 1, -1)
    done = torch.zeros(size, device=source.device, dtype=torch.bool)
    result = []
    past = None
    for i in range(limit + 1):
        hidden, past = writer.decode(x, context, valid, past, i)
        natural, bag = (
            (hidden[:, 0] @ keys.T).float() / math.sqrt(hidden.shape[-1])
        ).chunk(2)
        admissible = natural >= natural.amax(1, keepdim=True) - math.log(4.0)
        # The two log partition functions are constant across candidate cells.
        action = (2 * natural - bag).masked_fill(~admissible, -torch.inf).argmax(1)
        ids = inventory[action]
        result.append(ids.masked_fill(done, 0))
        done = done | ids.eq(1)
        if done.all():
            break
        x = bank[ids].repeat(2, 1)[:, None]
    return torch.stack(result, 1)


def bitmap_ranks(tiles, device):
    by_bytes = {v: i for i, v in enumerate(sorted(set(map(bytes, tiles))))}
    return torch.tensor([by_bytes[bytes(t)] for t in tiles], device=device)
