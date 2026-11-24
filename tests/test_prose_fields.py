import torch
from ilm.visual_lm.prose_fields import endpoints, interval_loss, select


def test_partition_gradient_and_decoded_span_match_raw_interval_enumeration():
    torch.manual_seed(953)
    raw = [torch.randn(2, 3, 6, requires_grad=True) for _ in range(3)]
    null = torch.randn(2, 3, requires_grad=True)
    length = torch.tensor([4, 6])
    truth = torch.tensor([[[0, 2], [-1, -1], [2, 4]], [[5, 6], [0, 6], [-1, -1]]])
    pred = (*endpoints(*raw), null)
    loss = interval_loss(pred, length, truth)
    actual = select(pred, length)
    losses = []
    for u, n in enumerate(length.tolist()):
        for f in range(3):
            spans = [(-1, -1)] + [(i, j) for i in range(n) for j in range(i + 1, n + 1)]
            scores = torch.stack(
                [null[u, f]]
                + [
                    raw[0][u, f, i] + raw[1][u, f, j - 1] + raw[2][u, f, i:j].sum()
                    for i, j in spans[1:]
                ]
            )
            losses.append(
                scores.logsumexp(0) - scores[spans.index(tuple(truth[u, f].tolist()))]
            )
            assert actual[u, f].tolist() == list(spans[scores.argmax()])
    exact = torch.stack(losses).mean()
    torch.testing.assert_close(loss, exact)
    got = torch.autograd.grad(loss, [*raw, null], retain_graph=True)
    for a, b in zip(got, torch.autograd.grad(exact, [*raw, null])):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)


def test_public_name_change_only_changes_the_requested_rating():
    import numpy as np
    from scripts.prepare_prose_fields import WEIGHTS
    from scripts.write_prose_fields import ProseReply, load_png

    torch.set_num_threads(2)
    net = ProseReply(arm="entity-ink")
    folder = WEIGHTS / "examples"
    pages = [load_png(folder / f"restaurant-{n}-page.png") for n in (1, 2)]
    assert np.array_equal(pages[0][[0, 1, 3]], pages[1][[0, 1, 3]])
    assert not np.array_equal(pages[0][2], pages[1][2])
    for i, out in enumerate(net.batch(pages), 1):
        assert out["keys"] == ["评分"] and not out["missing"]
        assert (
            out["pixels"].tobytes()
            == (folder / f"restaurant-{i}-answer.bin").read_bytes()
        )
    assert (folder / "restaurant-1-answer.bin").read_bytes() != (
        folder / "restaurant-2-answer.bin"
    ).read_bytes()


def test_name_matching_uses_bitmap_equality_and_rejects_overlapping_ambiguity():
    import numpy as np
    from ilm.visual_lm.prose_fields import name_position, pictured_name

    a, b, c = np.random.default_rng(65).integers(0, 256, (3, 128), dtype=np.uint8)
    assert pictured_name(np.stack((a, b, c, b)), np.stack((b, c))) == 1
    assert pictured_name(np.stack((a, a, a)), np.stack((a, a))) is None
    assert not name_position(10, None, 2).any()
    flags = name_position(5, 1, 2)
    np.testing.assert_array_equal(flags[:, 0], [0, 1, 1, 0, 0])
