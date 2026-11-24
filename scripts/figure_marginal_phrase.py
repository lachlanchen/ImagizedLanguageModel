"""Publish original held answer pixels and measured development snapshots."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scripts.figure_phrase_region import figure
from scripts.figure_recurrent_evidence import FIG
from scripts.prepare_marginal_confirmation import PUB, WORK
from scripts.prepare_open_exam import read
from scripts.train_marginal_phrase import MODEL


def trajectory():
    snapshots = read(PUB / "trajectory.json")
    steps = sorted(int(k) for k, v in snapshots.items() if v is not None)
    missing = sorted(int(k) for k, v in snapshots.items() if v is None)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3), sharey=True)
    for panel, (ax, domain, title) in enumerate(zip(
        axes, ("statements", "exam"), ("English statements", "Chinese reading questions")
    )):
        scores = [snapshots[str(step)]["scores"][domain] for step in steps]
        assert all(s["n"] == 192 for s in scores)
        for metric, label, marker, color in (
            ("fact", "Requested fact present", "o", "#00736d"),
            ("bitmap_exact", "Exact answer image", "s", "#9a4e32"),
        ):
            vals = [s[metric] for s in scores]
            ax.scatter(steps, vals, color=color, marker=marker, s=30, label=label, clip_on=False, zorder=3)
            for x in range(1, len(steps)):
                if not any(steps[x-1] < m < steps[x] for m in missing):
                    ax.plot(steps[x-1:x+1], vals[x-1:x+1], color=color, linewidth=1.4)
        for x in missing:
            ax.axvline(x, linestyle=":", color="#9aaba5", linewidth=1)
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=11)
        ax.text(-.065, 1.07, "ab"[panel], transform=ax.transAxes, fontweight="bold", fontsize=13)
        ax.set_xlim(-270, 10200)
        ax.set_ylim(-2, 48)
        ax.set_xticks([0,2400,4800,7200,9912], ["0","2,400","4,800","7,200","9,912"])
        ax.set_xlabel("Optimizer updates")
        ax.tick_params(labelsize=8)
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y", color="#dce5e1", linewidth=.6)
    axes[0].set_ylabel("Answers (of 192)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="lower center", frameon=False, fontsize=9)
    fig.subplots_adjust(left=.095,right=.975,bottom=.29,top=.82,wspace=.20)
    for ext in ("png","pdf"):
        fig.savefig(FIG / ("Figure_167_learning_from_pictured_answers."+ext), dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    figure(WORK, PUB, MODEL, "marginal", "new", "Figure_166_learning_complete_answers", "All paths")
    trajectory()
