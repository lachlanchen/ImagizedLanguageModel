# Qualification cache API recovery

The first full-cache attempt stopped before writer training. It saved all
27,532 training records and 4,072,494 target-predicting states, then failed on
the first training qualification case: the cache loader returned three values
and that caller expected two. The repair changes that caller to unpack the
states, physical patterns and record. It leaves the model, data, initialization,
normalization rule, optimizer, schedule and acceptance criteria unchanged.

The [recovery record](evidence/joint_writer_cache_tuple_recovery_20261113.json)
preserves the original registration and source hashes, failed execution cost
and recovery limits. The original attempt's logs, source snapshots and
normalization are retained privately. A CPU integrity pass authenticated every
saved training array and checked the bound target pixels for all 48 reused
training qualification cases. It reproduced the old unpack error on the real
cache and verified the three-value interface. Eight existing unit checks passed.
These checks establish cache reuse, not writing quality or language ability.

The failed cache child consumed 39,341.924 seconds at 18.302 GiB peak allocated
VRAM. CPU verification consumed 136.636 seconds at 0.651 GiB peak RSS, with no
model calls or CUDA initialization. Its measured time is also debited to the
original cumulative cache budget. No stage or pressure allowance is reset.

The amended registration must be committed and pushed before one explicit
coordinator resume. The cache path authenticates and reuses saved arrays;
remaining qualification preparation, the full fit, audit and output review
still have to run. Consult the private handoff for current ownership before
starting anything. The new writer's physical and autonomous language results
remain unmeasured.
