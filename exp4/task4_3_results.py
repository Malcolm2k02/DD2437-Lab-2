"""Plots, diagnostics and saved outputs for task 4.3."""

import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PARTIES = ["No party", "m", "fp", "s", "v", "mp", "kd", "c"]
COLORS = ["#777777", "#2355a4", "#55aadd", "#e44343", "#8b183a",
          "#58a44b", "#664a9c", "#d5a91c"]


def within_cell_agreement(bmu, labels):
    """Fraction of unordered co-located MP pairs with the same label."""
    total = same = 0
    for cell in np.unique(bmu):
        group = labels[bmu == cell]
        counts = np.unique(group, return_counts=True)[1]
        total += len(group) * (len(group) - 1) // 2
        same += int(np.sum(counts * (counts - 1) // 2))
    return same / total if total else 0.0


def diagnostics(votes, labels, weights, bmu, history, seed):
    distances = np.sum((votes[:, None, :] - weights[None, :, :]) ** 2, axis=2)
    nearest = np.argsort(distances, axis=1, kind="stable")[:, :2]
    coordinates = np.column_stack(np.unravel_index(np.arange(100), (10, 10)))
    separation = np.max(np.abs(coordinates[nearest[:, 0]]
                               - coordinates[nearest[:, 1]]), axis=1)
    result = {"seed": seed, "epochs": len(history),
              "quantization_error": float(history[-1]),
              "topographic_error_8_neighbours": float(np.mean(separation > 1)),
              "occupied_cells": int(len(np.unique(bmu))),
              "missing_vote_fraction": float(np.mean(votes == 0.5)),
              "label_agreement": {}}
    rng = np.random.default_rng(seed + 1)
    for key, values in labels.items():
        observed = within_cell_agreement(bmu, values)
        null = np.array([within_cell_agreement(bmu, rng.permutation(values))
                         for _ in range(500)])
        result["label_agreement"][key] = {
            "observed": observed, "shuffled_mean": float(null.mean()),
            "shuffled_95_percent_interval": np.quantile(null, [0.025, 0.975]).tolist(),
            "permutation_p_ge_observed": float((1 + np.sum(null >= observed)) / 501)}
    means = np.array([votes[labels["party"] == p].mean(axis=0) for p in range(8)])
    result["party_mean_vote_distances"] = np.linalg.norm(
        means[:, None, :] - means[None, :, :], axis=2).tolist()
    return result


def display_positions(bmu):
    """Deterministic subcell packing displays every MP without overplotting."""
    positions = np.empty((len(bmu), 2))
    for cell in np.unique(bmu):
        indices = np.flatnonzero(bmu == cell)
        width = int(np.ceil(np.sqrt(len(indices))))
        rows = int(np.ceil(len(indices) / width))
        positions[indices, 0] = cell % 10 + (np.arange(len(indices)) % width + .5) / width - .5
        positions[indices, 1] = cell // 10 + (np.arange(len(indices)) // width + .5) / rows - .5
    return positions


def grid_axes(ax):
    ax.set(xlim=(-.5, 9.5), ylim=(9.5, -.5), xticks=range(10), yticks=range(10))
    ax.set_xticks(np.arange(-.5, 10), minor=True)
    ax.set_yticks(np.arange(-.5, 10), minor=True)
    ax.grid(which="minor", color="#dddddd", linewidth=.5)
    ax.tick_params(which="minor", length=0)
    ax.set_aspect("equal")


def save_plots(votes, labels, bmu, history, out):
    positions = display_positions(bmu)
    occupancy = np.bincount(bmu, minlength=100)[bmu]
    sizes = np.minimum(17, 650 / occupancy)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, key, names, colors in [(axes[0], "party", PARTIES, COLORS),
                                   (axes[1], "gender", ["Male", "Female"],
                                    ["#277da8", "#d05b87"])]:
        for code, (name, color) in enumerate(zip(names, colors)):
            selected = labels[key] == code
            ax.scatter(*positions[selected].T, s=sizes[selected], c=color,
                       label=f"{name} (n={selected.sum()})", edgecolors="none")
        grid_axes(ax)
        ax.set_title(key.capitalize())
        ax.legend(loc="upper center", bbox_to_anchor=(.5, -.07), ncol=4, fontsize=8)
    fig.suptitle("MP voting SOM: same map, one dot per MP\nOffsets inside cells are for visibility only")
    fig.tight_layout(rect=(0, .07, 1, .93))
    fig.savefig(out / "party_gender.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(5, 6, figsize=(15, 13))
    for district, ax in zip(range(1, 30), axes.flat):
        selected = labels["district"] == district
        ax.scatter(*positions.T, s=np.minimum(3, 40 / occupancy), c="#dedede")
        ax.scatter(*positions[selected].T, s=np.minimum(13, 100 / occupancy[selected]),
                   c="#225e91", edgecolors="none")
        grid_axes(ax)
        ax.set_title(f"District {district} (n={selected.sum()})", fontsize=9)
        ax.tick_params(labelsize=6)
    axes.flat[-1].axis("off")
    fig.suptitle("District membership on the same voting SOM\nBlue: district MPs; grey: all MPs")
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(out / "districts.png", dpi=160)
    plt.close(fig)

    counts = np.bincount(bmu, minlength=100)
    missing = np.full(100, np.nan)
    for cell in np.unique(bmu):
        missing[cell] = np.mean(votes[bmu == cell] == .5)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, values, title, cmap in [(axes[0], counts, "MPs per cell", "viridis"),
                                    (axes[1], missing, "Missing-vote fraction", "magma")]:
        im = ax.imshow(values.reshape(10, 10), cmap=cmap)
        grid_axes(ax)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=.8)
    axes[2].plot(np.arange(1, len(history) + 1), history)
    axes[2].set(xlabel="Epoch", ylabel="Mean Euclidean distance to BMU",
                title="Quantization error")
    fig.tight_layout()
    fig.savefig(out / "diagnostics.png", dpi=160)
    plt.close(fig)


def save_results(votes, labels, names, weights, bmu, history, out, seed):
    out.mkdir(parents=True, exist_ok=True)
    save_plots(votes, labels, bmu, history, out)
    result = diagnostics(votes, labels, weights, bmu, history, seed)
    (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    np.savez(out / "som.npz", weights=weights.reshape(10, 10, 31),
             bmu=bmu, quantization_error=history)
    with (out / "mp_assignments.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["name", "row", "column", "party", "gender", "district"])
        for i, name in enumerate(names):
            writer.writerow([name, bmu[i] // 10, bmu[i] % 10,
                             PARTIES[labels["party"][i]],
                             ["Male", "Female"][labels["gender"][i]],
                             labels["district"][i]])
    print(json.dumps(result, indent=2))
    print(f"Saved plots, metrics, assignments and weights to {out}")
