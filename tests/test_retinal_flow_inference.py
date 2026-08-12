from __future__ import annotations

import pytest

from scripts.infer_retinal_flow_lm import summarize_trajectory


def test_summarize_trajectory_measures_closed_loop_ink_survival() -> None:
    traces = [
        {
            "mean_ink": value,
            "binary_ink_fraction": value,
            "candidate_mean_cosine": value / 2,
            "selected_energy": 1.0 - value,
        }
        for value in (0.4, 0.3, 0.2, 0.1)
    ]
    summary = summarize_trajectory(traces, window=2)

    assert summary["first_window_mean_ink"] == pytest.approx(0.35)
    assert summary["last_window_mean_ink"] == pytest.approx(0.15)
    assert summary["late_to_early_ink_ratio"] == pytest.approx(3 / 7)
    assert summary["sparse_cell_fraction"] == 0.0
    assert summary["window_cells"] == 2.0


def test_summarize_trajectory_accepts_empty_trace() -> None:
    assert summarize_trajectory([]) == {}
