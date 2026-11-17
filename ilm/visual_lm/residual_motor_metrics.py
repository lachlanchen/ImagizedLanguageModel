"""Keep actual generated work distinct from remaining image-error complexity."""
from .binary_codec_metrics import codec_metrics


def residual_motor_metrics(output, target, partition, remaining_events, generated_events, code_bits=256):
    reports = codec_metrics(output, target, partition, remaining_events, 'continuous')
    selections = dict(all=partition['held'] | ~partition['held'], train=~partition['held'], held=partition['held'])
    for name, report in reports.items():
        report['mean_remaining_error_tree_events'] = report.pop('mean_correction_tree_events')
        report['mean_generated_correction_events'] = float(generated_events[selections[name]].double().mean())
        report['mean_plan_and_generated_events'] = code_bits // 8 + report['mean_generated_correction_events']
    return reports
