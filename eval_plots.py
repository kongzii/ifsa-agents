import pandas as pd
import matplotlib.pyplot as plt

# Create the data dictionary.
data = {
    "Agents": [
        "OpenaiSearchAgentHigh",
        "OpenaiSearchAgentVariable (reasoning=low, search_context_size=low)",
        "OpenaiSearchAgentVariable (reasoning=low, search_context_size=medium)",
        "OpenaiSearchAgentVariable (reasoning=low, search_context_size=high)",
        "OpenaiSearchAgentVariable (reasoning=medium, search_context_size=low)",
        "OpenaiSearchAgentVariable (reasoning=medium, search_context_size=medium)",
        "OpenaiSearchAgentVariable (reasoning=medium, search_context_size=high)",
        "OpenaiSearchAgentVariable (reasoning=high, search_context_size=low)",
        "OpenaiSearchAgentVariable (reasoning=high, search_context_size=medium)",
        "OpenaiSearchAgentVariable (reasoning=high, search_context_size=high)"
    ],
    "MSE for p_yes": [0.0972761, 0.111357, 0.118189, 0.111162, 0.118585, 0.120755, 0.111844, 0.117988, 0.115736, 0.107958],
    "Mean confidence": [0.645455, 0.7425, 0.7275, 0.7175, 0.7165, 0.6625, 0.7195, 0.645, 0.685, 0.665],
    "% within +-0.05": [18.1818, 15, 15, 10, 25, 25, 30, 15, 15, 15],
    "% within +-0.1": [18.1818, 35, 25, 35, 40, 45, 50, 45, 35, 50],
    "% within +-0.2": [27.2727, 55, 50, 55, 55, 55, 55, 65, 50, 65],
    "% correct outcome": [45.4545, 65, 60, 70, 65, 65, 65, 60, 55, 70],
    "% precision for yes": [50, 71.4286, 66.6667, 83.3333, 71.4286, 71.4286, 71.4286, 66.6667, 57.1429, 83.3333],
    "% precision for no": [42.8571, 61.5385, 57.1429, 64.2857, 61.5385, 61.5385, 61.5385, 57.1429, 53.8462, 64.2857],
    "% recall for yes": [33.3333, 50, 40, 50, 50, 50, 50, 40, 40, 50],
    "% recall for no": [60, 80, 80, 90, 80, 80, 80, 80, 70, 90],
    "confidence/p_yes error correlation": [-0.0931103, 0.237798, 0.370676, -0.149673, 0.366815, -0.253487, 0.251514, 0.0751594, 0.271285, 0.132248],
    # We'll skip "Mean info_utility", "Mean cost ($)", and "Mean time (s)" since they are empty.
    "Proportion answerable": [1]*10,
    "Proportion answered": [0.55, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}

# Create a DataFrame.
df = pd.DataFrame(data)

# Choose a subset of metrics to plot (skip those with constant/empty values).
metrics = [
    "MSE for p_yes", "Mean confidence", "% within +-0.05", "% within +-0.1",
    "% within +-0.2", "% correct outcome", "% precision for yes", "% precision for no",
    "% recall for yes", "% recall for no", "confidence/p_yes error correlation", "Proportion answered"
]

# Set up a grid of subplots.
n_metrics = len(metrics)
ncols = 3
nrows = (n_metrics + ncols - 1) // ncols  # round up
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, nrows * 3))

# Flatten axes for easier iteration in case of multiple rows.
axes = axes.flatten()

for i, metric in enumerate(metrics):
    ax = axes[i]
    ax.bar(df["Agents"], df[metric], color='skyblue')
    ax.set_title(metric)
    # Rotate x-axis labels for clarity.
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.set_ylabel(metric)
    ax.set_xlabel("Agent")
    
# If there are unused axes, remove them.
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
