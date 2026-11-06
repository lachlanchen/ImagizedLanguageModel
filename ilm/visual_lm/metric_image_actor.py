"""Load a completed image-language checkpoint without its training data or ancestors."""
from pathlib import Path

import torch

from .native_image_actor import build_native_image_actor
from .grounded_image_weights import file_sha
from .ragged_metric_image_continuation import RaggedMetricImageLanguageModel


WARM_SHA = "7da31d229f4c5cc5bb51a5f5776a2c6e61e444f171ba412de547165911cb532c"


def completed_metric_checkpoint(path):
    """Inspect metadata on CPU before any expensive model initialization."""
    saved = torch.load(Path(path),map_location="cpu",weights_only=True,mmap=True)
    identity = saved.get("identity",{})
    arm = saved.get("arm")
    if (identity.get("purpose")!="matched_state_only_decoder_metric"
            or identity.get("arms")!={"control":0.,"metric":.001}
            or identity.get("warm_checkpoint_sha256")!=WARM_SHA
            or identity.get("updates_per_arm")!=2000 or identity.get("batch_size")!=4
            or identity.get("training_examples")!=32 or arm not in ("control","metric")
            or saved.get("update")!=2000 or saved.get("training_complete") is not True):
        raise ValueError("a completed declared 2,000-update V82 checkpoint is required")
    return saved


def load_metric_image_actor(*,checkpoint,controls,codec,ollama_root,gguf_source,device):
    """Materialize the fixed image/context body plus this final learned state.

    The warm hash records ancestry; its file is never opened. No training
    dataset, answer cache, vocabulary or teacher is needed by this loader.
    Completion metadata does not establish language quality.
    """
    checkpoint = Path(checkpoint)
    saved = completed_metric_checkpoint(checkpoint)
    state,identity,arm = saved["model"],saved["identity"],saved["arm"]
    actor,provenance = build_native_image_actor(controls=controls,codec=codec,
        ollama_root=ollama_root,gguf_source=gguf_source,device=device)
    if float(state["input_scale"])!=provenance["input_scale"]:
        raise ValueError("the checkpoint's physical input scale differs from its fixed source")
    writing = RaggedMetricImageLanguageModel(actor.writing.continuation.context,actor.writing.writer,
        input_scale=provenance["input_scale"],query_seed=20268103,metric_strength=identity["arms"][arm]).to(device)
    writing.load_trainable_state_dict(state);actor.writing = writing
    trained = sum(p.numel() for p in writing.parameters() if p.requires_grad)
    complete = sum(p.numel() for p in actor.parameters())
    if trained!=53096961 or complete!=7583225971:
        raise ValueError("checkpoint architecture differs from the actual matched actor")
    actor.requires_grad_(False).eval()
    return actor,dict(**provenance,checkpoint_sha256=file_sha(checkpoint),training_identity=identity,
        arm=arm,complete_actor_parameters=complete,trained_parameters=trained,
        ancestor_checkpoints_opened=False,training_dataset_loaded=False,
        quality_qualification="not assessed by checkpoint completion metadata",language_quality_claim=False)
