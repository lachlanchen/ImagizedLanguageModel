from __future__ import annotations

from .canonical_glyph_language import canonical_glyph_language_config_from_payload
from .exact_raster_density_v54 import ExactRasterDensity, ExactRasterDensityConfig
from .hierarchical_raster_language import CausalRasterReader

ARCHITECTURE = "joint-exact-raster-language-v55"


class JointExactRasterLanguage(ExactRasterDensity):
    """V54's exact image law, permitting a likelihood gradient into its reader."""

    def __init__(self, reader, config, *, reader_trainable):
        super().__init__(reader, config)
        self.reader_jointly_trained = bool(reader_trainable)
        self.reader.requires_grad_(reader_trainable)
        # This scalar belongs to the earlier contrastive objective, not NLL.
        self.reader.logit_scale.requires_grad_(False)

    def read(self, context):
        states = self.reader.language(context)
        return states["hidden_states"][:, -1].float(), states["anchor_fields"][
            :, -1
        ].float()

    def boundary_receipt(self):
        receipt = super().boundary_receipt()
        receipt.update(
            architecture=ARCHITECTURE,
            reader_jointly_trained=self.reader_jointly_trained,
            trainable_parameters=sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
            reader_dropout_during_training=False,
        )
        return receipt


def model_from_checkpoint(checkpoint, device, *, trainable=False):
    reader = CausalRasterReader(
        canonical_glyph_language_config_from_payload(checkpoint["reader_config"])
    )
    model = JointExactRasterLanguage(
        reader,
        ExactRasterDensityConfig(**checkpoint["model_config"]),
        reader_trainable=checkpoint.get("arm") == "joint",
    )
    model.load_state_dict(
        checkpoint.get("ema_model", checkpoint.get("model")), strict=True
    )
    model.to(device)
    if not trainable:
        model.requires_grad_(False)
    return model.eval()
