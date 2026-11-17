"""CPU physical metrics for every retained codec image; no recognizer."""
import torch

from .binary_plan_glyph_density import binary_images
from scripts.measure_reader_writing_consequence import pixels


def codec_metrics(reconstruction,target,partition,events,mode):
    a,b=binary_images(reconstruction).bool(),binary_images(target).bool()
    if (a.shape!=b.shape or events.shape!=(len(a),) or mode not in ('binary','continuous')
            or not bool(((events>=1)&(events<=342)&(events==events.long())).all())):
        raise ValueError('Every source image and valid physical event count required')
    held,group=partition['held'],partition['group'];reports={}
    if held.shape!=(len(a),) or group.shape!=(len(a),):raise ValueError('Complete source partition required')
    tp=(a&b).flatten(1).sum(1).double();fp=(a&~b).flatten(1).sum(1).double();fn=(~a&b).flatten(1).sum(1).double()
    denominator=2*tp+fp+fn
    f1=torch.where(denominator>0,2*tp/denominator.clamp_min(1),torch.ones_like(tp))
    exact=(a==b).flatten(1).all(1).double()
    for name,selection in [('all',torch.ones(len(a),dtype=torch.bool)),('train',~held),('held',held)]:
        indices=selection.nonzero().flatten();g=group[indices]
        counts=torch.bincount(g,minlength=len(partition['group_mask_sha256']));present=counts>0
        def macro(value):
            sums=torch.zeros(len(counts),dtype=torch.float64);sums.scatter_add_(0,g,value[indices])
            return float((sums[present]/counts[present]).mean())
        report=dict(pixels=pixels(a[indices],b[indices]),distinct_forms=int(present.sum()),
                    macro_form_f1=macro(f1),macro_form_exact_rate=macro(exact),
                    mean_correction_tree_events=float(events[indices].double().mean()))
        if mode=='binary':report['mean_plan_and_correction_events']=8+report['mean_correction_tree_events']
        reports[name]=report
    return reports
