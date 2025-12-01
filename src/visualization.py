"""
Visualization utilities for experiment results.
"""

import matplotlib.pyplot as plt
import numpy as np


def create_performance_plot(summary_data: list[dict]) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Extract data for plotting
    experiments = [exp["experiment"] for exp in summary_data]
    test_accuracies = [exp["test_accuracy"] for exp in summary_data]
    val_accuracies = [exp["best_val_accuracy"] for exp in summary_data]
    transforms = [exp["transform"] for exp in summary_data]

    # Colors based on transform type
    transform_colors = {
        "none": "skyblue",
        "light": "lightgreen",
        "medium": "gold",
        "heavy": "lightcoral",
    }
    colors = [transform_colors.get(t, "gray") for t in transforms]

    # Plot 1: Test vs Validation Accuracy
    x_pos = np.arange(len(experiments))
    width = 0.35

    bars1 = ax1.bar(
        x_pos - width / 2,
        test_accuracies,
        width,
        label="Test Accuracy",
        color=colors,
        alpha=0.8,
    )
    bars2 = ax1.bar(
        x_pos + width / 2,
        val_accuracies,
        width,
        label="Val Accuracy",
        color=colors,
        alpha=0.6,
    )

    ax1.set_xlabel("Experiments")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Test vs Validation Accuracy by Experiment")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(experiments, rotation=45, ha="right")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Plot 2: Transform comparison
    transform_accuracies = {}
    for exp in summary_data:
        transform = exp["transform"]
        if transform not in transform_accuracies:
            transform_accuracies[transform] = []
        transform_accuracies[transform].append(exp["test_accuracy"])

    transform_means = {t: np.mean(accs) for t, accs in transform_accuracies.items()}
    transform_stds = {t: np.std(accs) for t, accs in transform_accuracies.items()}

    transform_names = list(transform_means.keys())
    transform_vals = [transform_means[t] for t in transform_names]
    transform_errs = [transform_stds[t] for t in transform_names]
    transform_colors_plot = [transform_colors.get(t, "gray") for t in transform_names]

    bars = ax2.bar(
        transform_names,
        transform_vals,
        yerr=transform_errs,
        capsize=5,
        color=transform_colors_plot,
        alpha=0.8,
    )
    ax2.set_xlabel("Transform Type")
    ax2.set_ylabel("Test Accuracy")
    ax2.set_title("Performance by Transform Type")
    ax2.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig("experiment_performance_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()


def print_experiment_summary(summary_data: list[dict]) -> None:
    print("\n" + "=" * 120)
    print(" EXPERIMENT SUMMARY REPORT")
    print("=" * 120)

    # Header
    header = f"{'Experiment':<20} {'Transform':<10} {'LR':<12} {'Test Acc':<10} {'Test F1':<10} {'Val Acc':<10} {'Train Acc':<10}"
    print(header)
    print("-" * len(header))

    # Data rows
    for exp in summary_data:
        row = (
            f"{exp['experiment']:<20} "
            f"{exp['transform']:<10} "
            f"{exp['learning_rate']:<12.2e} "
            f"{exp['test_accuracy']:<10.4f} "
            f"{exp['test_f1']:<10.4f} "
            f"{exp['best_val_accuracy']:<10.4f} "
            f"{exp['final_train_accuracy']:<10.4f}"
        )
        print(row)

    # Key insights
    best_exp = summary_data[0]
    worst_exp = summary_data[-1]

    print("\n" + "=" * 120)
    print(" KEY INSIGHTS")
    print("=" * 120)

    print(f" BEST PERFORMER: {best_exp['experiment']}")
    print(f"   Test Accuracy: {best_exp['test_accuracy']:.4f}")
    print(f"   Transform: {best_exp['transform']}")
    print(f"   Learning Rate: {best_exp['learning_rate']:.2e}")

    print(f"\n WORST PERFORMER: {worst_exp['experiment']}")
    print(f"   Test Accuracy: {worst_exp['test_accuracy']:.4f}")
    print(f"   Transform: {worst_exp['transform']}")

    # Transform analysis
    print("\n TRANSFORM ANALYSIS")
    print("-" * 50)
    transform_stats = {}
    for exp in summary_data:
        transform = exp["transform"]
        if transform not in transform_stats:
            transform_stats[transform] = []
        transform_stats[transform].append(exp["test_accuracy"])

    for transform, accuracies in transform_stats.items():
        avg_acc = np.mean(accuracies)
        best_acc = max(accuracies)
        print(f"   {transform:8}: Avg = {avg_acc:.4f} (Best = {best_acc:.4f})")

    all_accuracies = [exp["test_accuracy"] for exp in summary_data]
    print(
        f"\n PERFORMANCE RANGE: {min(all_accuracies):.4f} - {max(all_accuracies):.4f}"
    )
    print(
        f" AVERAGE ACCURACY:  {np.mean(all_accuracies):.4f} ± {np.std(all_accuracies):.4f}"
    )
