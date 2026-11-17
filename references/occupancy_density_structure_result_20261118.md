# Lossless occupancy support and measured event cost

The CPU structure audit recovers every original bit of all **68,948 cells**
in the 29,740 distinct saved training-span images. Both modes preserve the
original packed bytes, canonical prefix-determined topology and all 3,206
empty cells. No model forward, optimizer update, new target rendering or GPU
call occurs.

| Physical factorization | Mean events/cell | Maximum observed | Component parameters |
|---|---:|---:|---:|
| Flat, same learned root | 245.096304 | 257 | 2,261,074 |
| Occupancy tree | 97.302489 | 193 | 2,261,074 |

The tree uses .396996964 times as many events on this exact saved support.
This is a representation count, not measured neural throughput, compression
in bytes or language-training efficiency. The complete binary support includes
denser images than observed here; its maximum is 342 events. These saved cells
are not corpus exposures, target-only counts or held-out examples.

The audit takes **6.521 seconds** with two CPU threads, 657,690,624-byte peak
process RSS and 54.781 GiB minimum sampled available host RAM. Eight preceding
CPU checks cover exhaustive 4-by-4 losslessness, actual small-model probability
normalization, causal/cache replay, generated support, production-size cases,
malformed topology rejection, finite gradients and exact trial reset.

The [bound result](evidence/occupancy_density_structure_result_20261118.json)
retains the full event-length and depth/mask counts. The
[design](occupancy_visual_density_direction_20261118.md) states the normalized
image law and its prior work. No neural fit or usable language follows from
the structural pass. The next test must measure actual conditional writing
with the same real source states and unchanged language criteria.
