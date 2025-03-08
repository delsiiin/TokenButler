import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv("reasoning_result.csv")

# Define the metrics you want to plot
metrics = ["Perplexity", "BBH Causal Judgement", "MMLU-Pro"]

# Create a figure with 3 subplots in a single row
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
xid = 0
for ax, metric in zip(axes, metrics):
    xid += 1
    # Plot Dense as a horizontal line
    dense_val = df.loc[df["Method"] == "Dense", metric].values[0]
    ax.axhline(dense_val, linestyle="--", linewidth=1, label="Dense", color="black")

    # Plot Oracle and TokenButler as lines
    for method in ["Oracle", "TokenButler"]:
        subset = df[df["Method"] == method].sort_values(by="Sparsity (%)")
        ax.plot(
            subset["Sparsity (%)"],
            subset[metric],
            marker="o",
            linestyle="-",
            linewidth=3,
            label=method
        )

    # Axis and label formatting
    ax.set_title(metric, fontsize=24)
    if xid == 2:
        ax.set_xlabel("Sparsity (%)", fontsize=22)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.tick_params(axis="x", labelsize=22)
    ax.tick_params(axis="y", labelsize=22)

# Collect handles/labels from the last subplot and place them in a single legend
handles, labels = axes[-1].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    fontsize=20,
    ncol=len(labels),
    bbox_to_anchor=(0.5, 1.0)
)

# Adjust layout so the legend does not overlap
plt.tight_layout(rect=[0, 0, 1, 0.87])

# Save to PDF
output_file = "reasoning_results_plots.pdf"
plt.savefig(output_file)
plt.close()
