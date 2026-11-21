"""Write descriptive plots and Markdown from computed evaluation results."""

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_report(output, metadata, comparison, configs, heldout, errors, labels, scores):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    for ax, (name, cfg) in zip(axes, ((n, c) for n, c in configs.items() if c["threshold_curve"]),
                               strict=True):
        curve = cfg["threshold_curve"]
        x = [row["threshold"] for row in curve]
        for metric in ("recall", "precision"):
            ax.plot(x, [row[metric] for row in curve], label=metric)
        ax.axvline(cfg["threshold"], color="gray", linestyle=":")
        ax.set(title=name.replace("_", " "), xlabel="Threshold", ylim=(0, 1.04))
    axes[0].set_ylabel("Training OOF metric")
    axes[-1].legend()
    save_figure(fig, output / "thresholds.png")

    fig, ax = plt.subplots(figsize=(4.7, 4))
    ConfusionMatrixDisplay(np.array(heldout["confusion_matrix"]), display_labels=[0, 1]).plot(
        ax=ax, cmap="Blues", colorbar=False,
    )
    ax.set_title("Final holdout")
    save_figure(fig, output / "confusion-matrix.png")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    RocCurveDisplay.from_predictions(labels, scores, ax=axes[0])
    PrecisionRecallDisplay.from_predictions(labels, scores, ax=axes[1])
    CalibrationDisplay.from_predictions(labels, scores, n_bins=5, strategy="quantile", ax=axes[2])
    axes[2].set_title("Descriptive calibration")
    save_figure(fig, output / "holdout-curves.png")

    measures = ["accuracy", "precision", "recall", "specificity", "f1", "f2",
                "roc_auc", "average_precision", "fp", "fn"]
    cells = ["Model", *measures]
    lines = [
        "# Cleveland model evaluation",
        "",
        "## Training-only nested comparison",
        "",
        "Values are outer-fold means ± sample standard deviations. FP and FN are mean counts per fold.",
        "",
        "| " + " | ".join(cells) + " |",
        "| " + " | ".join("---" for _ in cells) + " |",
    ]
    for name, result in comparison.items():
        summary = result["summary"]
        values = [f"{summary[k]['mean']:.3f} ± {summary[k]['sd']:.3f}" for k in measures]
        lines.append("| " + " | ".join([name, *values]) + " |")

    lines.extend(["", "## Full-training configurations", ""])
    for name, config in configs.items():
        lines.append(f"- **{name}** at threshold {config['threshold']:.2f} with `{config['parameters']}`.")

    lines.extend([
        "", "The threshold curves use out-of-fold training predictions after hyperparameter selection.",
        "The same inner folds informed tuning and threshold selection, so the curves are exploratory.",
        "Outer folds independently evaluate the combined procedure. Comparing outer means to choose",
        "a winner adds selection uncertainty. Fold variability describes these folds on this dataset.",
        "", "![Training threshold curves](thresholds.png)", "", "## Locked final model", "",
        f"Selected **{metadata['algorithm']}** at threshold **{metadata['threshold']:.2f}**.",
        "Selection uses mean outer F2 with a 0.01 simplicity margin. The always-positive control",
        "illustrates the precision and specificity cost of maximizing recall alone.",
        "", "## Final holdout", "", "| Metric | Value |", "| --- | ---: |",
    ])
    for name, value in heldout.items():
        if name != "confusion_matrix":
            lines.append(f"| {name} | {value:.4f} |" if isinstance(value, float)
                         else f"| {name} | {value} |")
    fp = [str(row["source_row"]) for row in errors if row["prediction"] == 1]
    fn = [str(row["source_row"]) for row in errors if row["prediction"] == 0]
    lines.extend([
        "", "![Final holdout confusion matrix](confusion-matrix.png)",
        "", "![ROC, precision-recall and calibration curves](holdout-curves.png)",
        "", "## Errors and limitations", "",
        f"- False-positive source rows are {', '.join(fp) or 'none'}.",
        f"- False-negative source rows are {', '.join(fn) or 'none'}.",
        "- Each misclassified positive changes recall by 1/28 and each negative changes specificity by 1/33.",
        "- Brier score and the calibration plot describe this small historical holdout. Clinical calibration,",
        "  diagnosis, and future cardiovascular risk prediction are outside the evidence of this experiment.",
        "", "See `selection.json` for the locked configuration, software versions and input schema,",
        "`validation.json` for tuning and fold results, and `heldout.json` for individual error scores.",
    ])
    (output / "evaluation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
