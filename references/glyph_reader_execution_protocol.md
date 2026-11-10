# Full-reader qualification and answer timing

This execution-only amendment is recorded before either fixed final candidate
produces qualification or assessment results. The source body, model outputs,
training recipe, cohort, stopping rule and acceptance thresholds stay unchanged.
Both conditions use the same monitor and timing instrumentation.

The original 48-GiB available-host entry requirement, 24-GiB runtime reserve,
22-GiB device allocation limit, strict low-pressure entry and project GPU mutex
remain in force. After large file-authentication reads, wait for low pressure
before strict admission. The full-reader monitor allows at most 1,800 seconds
of cumulative pressure waiting, matching the primary fit recovery. Its stage
deadline is also checked during every pressure-wait iteration. Qualification
retains the 3,600-second stage cap; assessment and standalone inference retain
their 1,800-second caps. A wait allowance does not extend those deadlines.
The original reader monitor and resource-admission helpers remain unchanged.

For every autonomous answer, synchronize the device immediately before and
after generation. Record wall time including image prefill, actual emitted-pixel
feedback and stopping, and record guard-measured pressure waiting within that
interval separately. Prompt file decoding, actor loading and output saving
are outside that generation interval. Actor loading is reported separately;
the aggregate stage receipt still accounts for the whole stage. Subtracting
recorded pressure waits does not isolate pure kernel time: CPU work, scheduling
and other execution overhead remain included.

Report complete-answer quality, failures and caps alongside timing for the
entire cohort. The scorer may report clear complete answers per measured
generation wall second; it must not drop failed answers to advertise speed.
No comparison with an unmeasured text-language baseline, foundation-training
efficiency advantage or general LLM parity follows from these timings.
