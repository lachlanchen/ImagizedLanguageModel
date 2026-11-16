"""Fixed image-feature metric for training-only content verification.

This changes the rejected bit-space verifier's metric, not its answer matcher.
It has no Unicode output or source-question input. Qualification remains external.
"""
import hashlib

import numpy as np

from .visual_content_verifier import SelectiveGlyphVerifier, packed_images


def accepted_in_feature_ball(distance, separation):
    distance = np.asarray(distance); separation = np.asarray(separation)
    if distance.shape != separation.shape or not np.isfinite(distance).all() or not np.isfinite(separation).all():
        raise ValueError('Finite aligned feature distances required')
    if np.any(distance < 0) or np.any(separation < 0):
        raise ValueError('Distances cannot be negative')
    return 2 * distance < separation


class FrozenEncoderVerifier(SelectiveGlyphVerifier):
    checkpoint = None
    last_instance = None

    def __init__(self, prototypes, *, guard=lambda: None):
        import torch
        from ilm.visual_lm.semantic_glyph_loop import SemanticGlyphBridge
        value = np.asarray(prototypes)
        if value.dtype != np.uint8 or value.ndim != 2 or value.shape[1] != 128 or len(value) < 2:
            raise ValueError('Complete physical source masks required')
        if len(np.unique(value,axis=0)) != len(value):
            raise ValueError('Keep one ID per distinct physical image')
        self.guard = guard
        self.prototypes = value.copy()
        self.identities = {row.tobytes():i for i,row in enumerate(value)}
        bridge = SemanticGlyphBridge()
        self.encoder = bridge.encoder
        del bridge
        payload = torch.load(self.checkpoint,weights_only=True,map_location='cpu',mmap=True)
        state = {k.removeprefix('encoder.'):v for k,v in payload['model'].items() if k.startswith('encoder.')}
        self.encoder.load_state_dict(state,strict=True)
        del state,payload
        self.encoder = self.encoder.requires_grad_(False).eval().to('cuda:0')
        self.encoder_parameters = sum(p.numel() for p in self.encoder.parameters())
        self.forward_calls = self.encoded_images = 0
        self.decode_traces = []
        self.features = self._features(np.unpackbits(value,axis=1,bitorder='big').reshape(-1,1,32,32))
        self.minimum_distance = np.empty(len(value),dtype=np.float64)
        with torch.no_grad():
            for first in range(0,len(value),64):
                self._guard()
                last = min(first+64,len(value))
                distance = torch.cdist(self.features[first:last],self.features,p=2,
                    compute_mode='donot_use_mm_for_euclid_dist')
                distance[torch.arange(last-first,device=distance.device),torch.arange(first,last,device=distance.device)] = float('inf')
                self.minimum_distance[first:last] = distance.min(1).values.cpu().numpy()
        digest = hashlib.sha256()
        for first in range(0,len(value),64):
            digest.update(self.features[first:first+64].cpu().numpy().tobytes())
        self.feature_sha256 = digest.hexdigest()
        type(self).last_instance = self

    def _guard(self):
        import torch
        self.guard()
        if torch.cuda.max_memory_allocated() > 2 * 1024**3:
            raise RuntimeError('Fixed metric exceeds 2 GiB allocated VRAM')

    def _features(self, ink):
        import torch
        result = []
        with torch.no_grad():
            for first in range(0,len(ink),64):
                self._guard()
                block = torch.from_numpy(ink[first:first+64].copy()).to('cuda:0',dtype=torch.float32)
                n = len(block)
                # Fixed batch shape avoids a change of encoder arithmetic at a tail.
                image = torch.ones(64,1,32,32,device='cuda:0')
                image[:n] = 1 - 2 * block
                feature = self.encoder(image).float()
                if not bool(torch.isfinite(feature).all()): raise ValueError('Nonfinite frozen image features')
                result.append(feature[:n].double())
                self.forward_calls += 1; self.encoded_images += n
        return torch.cat(result)

    def decode(self, ink, *, guard=lambda: None):
        import torch
        packed_images(ink)  # Preserve the original binary-domain contract.
        features = self._features(np.asarray(ink))
        ids = []; distances = []
        with torch.no_grad():
            for first in range(0,len(features),64):
                self._guard(); guard()
                distance = torch.cdist(features[first:first+64],self.features,p=2,
                    compute_mode='donot_use_mm_for_euclid_dist')
                d,index = distance.min(1)
                ids.extend(index.cpu().tolist()); distances.extend(d.cpu().tolist())
        ids = np.asarray(ids,dtype=np.int64); distances = np.asarray(distances,dtype=np.float64)
        accepted = accepted_in_feature_ball(distances,self.minimum_distance[ids])
        self.decode_traces.append(dict(cells=[hashlib.sha256(row.tobytes()).hexdigest() for row in packed_images(ink)],
            nearest_prototype_ids=ids.tolist(),distances=distances.tolist(),
            nearest_separations=self.minimum_distance[ids].tolist(),accepted=accepted.tolist()))
        return dict(identities=np.where(accepted,ids,-1),accepted=accepted,distances=distances,
                    nearest_prototype_ids=ids)
