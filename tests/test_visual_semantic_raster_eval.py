from __future__ import annotations

import inspect
from types import SimpleNamespace

import torch

from ilm.visual_lm.visual_semantic_raster_evaluation import RasterCharacterBank
from scripts.eval_visual_semantic_raster_v32 import (
    RasterEvaluationData,
    autonomous_generate_batch,
    controlled_prompts,
    target_reconstruction_generation,
    v32_gate_report,
)


def _summary(value: float) -> dict[str, object]:
    return {"mean": value, "count": 128, "ci95": [value, value]}


def _metrics(
    *,
    character_accuracy: float = 0.7,
    character_error_rate: float = 0.3,
    exact: float = 0.25,
    length_exact: float = 0.7,
    log_similarity: float = -0.5,
    nonblank: float = 1.0,
) -> dict[str, object]:
    return {
        "samples": 128,
        "finite_generation": True,
        "sequence": {
            "character_accuracy": _summary(character_accuracy),
            "character_error_rate": _summary(character_error_rate),
            "exact": _summary(exact),
            "length_exact": _summary(length_exact),
            "target_log_similarity": _summary(log_similarity),
        },
        "raster": {"nonblank_answer": _summary(nonblank)},
    }


def _data() -> RasterEvaluationData:
    return RasterEvaluationData(
        identifiers=("a", "b"),
        targets=("天", "地"),
        prompt_pixels=torch.stack(
            (torch.zeros(3, 16, 32), torch.ones(3, 16, 32))
        ),
        prompt_mask=torch.tensor([[1.0, 0.0], [1.0, 1.0]]),
        answer_cells=torch.zeros(2, 2, 1, 24, 24),
        answer_mask=torch.ones(2, 2),
        stop_targets=torch.zeros(2, 3),
        stop_mask=torch.ones(2, 3),
    )


def test_v32_autonomous_callable_cannot_receive_answers_or_candidate_bank() -> None:
    assert list(inspect.signature(autonomous_generate_batch).parameters) == [
        "model",
        "prompt_pixels",
        "prompt_mask",
    ]


def test_v32_prompt_controls_preserve_shapes_and_break_condition() -> None:
    data = _data()
    shuffled, shuffled_mask = controlled_prompts(data, "shuffled")
    blank, blank_mask = controlled_prompts(data, "blank")
    assert torch.equal(shuffled[0], data.prompt_pixels[1])
    assert torch.equal(shuffled_mask[0], data.prompt_mask[1])
    assert torch.equal(blank, torch.ones_like(blank))
    assert torch.equal(blank_mask, data.prompt_mask)


def test_v32_target_reconstruction_is_labeled_as_post_generation_diagnostic() -> None:
    data = _data()

    class ExactRasterAutoencoder:
        config = SimpleNamespace(maximum_answer_cells=2)

        def __call__(
            self,
            _prompt_pixels: torch.Tensor,
            _prompt_mask: torch.Tensor,
            answer_cells: torch.Tensor,
            _answer_mask: torch.Tensor,
            *,
            feedback_mode: str,
        ) -> SimpleNamespace:
            assert feedback_mode == "clean"
            bounded = answer_cells.clamp(0.01, 0.99)
            return SimpleNamespace(raster_logits=torch.logit(bounded))

    generated = target_reconstruction_generation(
        ExactRasterAutoencoder(),  # type: ignore[arg-type]
        data,
        device=torch.device("cpu"),
        precision="fp32",
        batch_size=1,
    )
    assert generated.finite
    assert torch.equal(generated.lengths, torch.tensor([2, 2]))
    assert torch.equal(generated.stop_probabilities[:, 2], torch.ones(2))
    assert torch.allclose(generated.cells, data.answer_cells.clamp(0.01, 0.99))


def test_v32_gate_report_never_claims_final_proof_without_ablations() -> None:
    correct = _metrics()
    shuffled = _metrics(log_similarity=-0.7, exact=0.1)
    blank = _metrics(exact=0.1)
    teacher = _metrics(character_accuracy=0.75)
    paraphrase = _metrics(character_accuracy=0.5)
    report = v32_gate_report(
        checkpoint_receipt={
            "finite_model_state": True,
            "boundary": {
                "parameter_cap_pass": True,
                "forbidden_parameter_names": [],
            },
        },
        correct=correct,
        shuffled=shuffled,
        blank=blank,
        teacher=teacher,
        paraphrase=paraphrase,
        training_font=_metrics(character_accuracy=0.75),
        frequency_baseline_exact=0.0,
        training_complete=True,
        peak_vram_bytes=10 * 1024**3,
    )
    assert report["integrity_pass"]
    assert report["direct_raster_pass"]
    assert report["available_visual_language_pass"]
    assert report["decision"] == "provisional-visual-semantic-raster-proof-pending-ablations"


def test_v32_candidate_bank_type_is_not_part_of_generate_signature() -> None:
    annotations = inspect.get_annotations(autonomous_generate_batch)
    assert RasterCharacterBank not in annotations.values()
    assert all("RasterCharacterBank" not in str(value) for value in annotations.values())
