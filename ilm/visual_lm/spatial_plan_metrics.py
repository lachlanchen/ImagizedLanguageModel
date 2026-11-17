"""Declare the actual number of physical plan events in codec measurements."""
from .binary_codec_metrics import codec_metrics


def plan_codec_metrics(reconstruction,target,partition,events,mode,code_bits):
    if type(code_bits) is not int or code_bits<8 or code_bits%8:raise ValueError('Byte-aligned physical plan width required')
    reports=codec_metrics(reconstruction,target,partition,events,mode)
    if mode=='binary':
        for report in reports.values():report['mean_plan_and_correction_events']=code_bits//8+report['mean_correction_tree_events']
    return reports
