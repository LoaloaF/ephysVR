import matplotlib.pyplot as plt
import numpy as np

labels = [
    "177x2 ch.",
    "100x2 ch.",
    "128 ch",
    "96 ch",
    "64 ch",
    "16 ch"
]

def label_to_channels(label):
    clean = label.replace("ch.", "").replace("ch", "").strip()
    if "x" in clean:
        left, right = clean.split("x", 1)
        return int(left) * int(right)
    return int(clean)

channel_counts = [label_to_channels(label) for label in labels]
channel_labels = [f"{count} ch." for count in channel_counts]

areas = [6336, 3600, 2560, 1920, 1280, 320]

x = np.array(channel_counts)

# Approximate rodent pyramidal neuron soma cross-sectional area
pyramidal_soma_area = 96  # µm²

plt.figure(figsize=(8, 5))
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.grid(True, which="both", axis="both", alpha=0.3)

# Connect only points 1-2
plt.plot(
    x[:2],
    areas[:2],
    marker="o",
    linewidth=2,
    color="red",
    label="Houman's shanks"
)

# Connect only points 3-4
plt.plot(
    x[2:4],
    areas[2:4],
    marker="o",
    linewidth=2,
    label="New shanks"
)

# Connect only points 5-6
plt.plot(
    x[4:],
    areas[4:],
    marker="o",
    linewidth=2,
    color="gray",
    label="Established shanks"
)

# Value labels
for xi, yi in zip(x, areas):
    plt.text(
        xi,
        yi + 100,
        f"{yi}",
        ha="center"
    )

# Pyramidal neuron reference
plt.axhline(
    pyramidal_soma_area,
    linestyle="--",
    linewidth=1.5,
    label="Pyramidal neuron soma (~96 µm²)"
)
ax.annotate(
    "Pyramidal neuron soma (~96 µm²)",
    xy=(x[-1], pyramidal_soma_area),
    xytext=(16, 8),
    textcoords="offset points",
    ha="left",
    va="bottom"
)

plt.xticks(x, channel_labels)
ax.annotate(
    "Houman's shanks",
    xy=(np.mean(x[:2]), np.mean(areas[:2])),
    xytext=(32, 8),
    textcoords="offset points",
    ha="left",
    va="bottom"
)
ax.annotate(
    "New shanks",
    xy=(np.mean(x[2:4]), np.mean(areas[2:4])),
    xytext=(32, 8),
    textcoords="offset points",
    ha="left",
    va="bottom"
)
ax.annotate(
    "Established shanks",
    xy=(np.mean(x[4:]), np.mean(areas[4:])),
    xytext=(32, 8),
    textcoords="offset points",
    ha="left",
    va="bottom"
)
plt.xlabel("n channels")
plt.ylabel("Cross section area (µm²)")
plt.title("Shank Cross Section Comparison")
plt.legend()

plt.tight_layout()
plt.savefig("shank_cross_sections.svg", format="svg")
plt.show()