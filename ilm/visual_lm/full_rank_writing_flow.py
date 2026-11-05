"""Full-dimensional continuous writing flow; no symbol inventory or text head."""

from torch import nn

from .semantic_writing_model import (
    JointWritingFlow,
    SemanticWritingConfig,
    SemanticWritingModel,
)


def full_rank_configuration():
    """The smallest hidden width without the old fixed output-span ceiling."""
    return SemanticWritingConfig(flow_width=1536)


class FullRankWritingFlow(JointWritingFlow):
    def __init__(self, config=None):
        config = full_rank_configuration() if config is None else config
        if config.flow_width < config.block_width:
            raise ValueError("the flow must preserve all noisy coordinate dimensions")
        super().__init__(config)
        # Preserve the noisy coordinates at initialization instead of learning
        # to invert an initially ill-conditioned square random projection.
        nn.init.eye_(self.noisy.weight)
        nn.init.zeros_(self.noisy.bias)


def build_full_rank_writer():
    """Same causal writer and checkpoint keys, with a full-dimensional flow."""
    config = full_rank_configuration()
    model = SemanticWritingModel(config)
    model.flow = FullRankWritingFlow(config)
    return model
